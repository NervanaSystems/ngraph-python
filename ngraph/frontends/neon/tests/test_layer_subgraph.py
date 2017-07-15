import pytest
import ngraph as ng
from ngraph.frontends.neon.layer import wrap_layer, LABELS, Layer
from ngraph.frontends.neon.utils import scope_ops, ComputationalGraph, SubGraph


class SimpleLayer(Layer):
    metadata = {"foo": "bar"}

    @wrap_layer
    def __call__(self, in_obj):
        if not self.initialized:
            w_axis = ng.make_axis()
            self.weight = ng.variable(axes=[w_axis],
                                      initial_value=2,
                                      metadata={"label": LABELS["weight"]},
                                      name="W")
            self.side_effect = ng.persistent_tensor(axes=[w_axis],
                                                    initial_value=0)

        return ng.sequential([ng.assign(self.side_effect, self.weight),
                              self.weight * in_obj])


class NestedLayer(Layer):

    def __init__(self, inner_layer=None, **kwargs):
        super(NestedLayer, self).__init__(**kwargs)
        if inner_layer is None:
            inner_layer = SimpleLayer(inherit_scope=self.scope)
        self.inner_layer = inner_layer

    @wrap_layer
    def __call__(self, in_obj):
        return self.inner_layer(in_obj)


@pytest.fixture(params=[32])
def batch_size(request):
    return request.param


@pytest.fixture(params=[4])
def input_size(request):
    return request.param


def test_layer_inputs(input_placeholder):
    """
    Test a simple layer determines inputs correctly
    """
    layer = SimpleLayer()
    assert isinstance(layer.inputs, dict)
    assert len(layer.inputs) == 0

    layer(input_placeholder)
    assert len(layer.inputs) == 1
    assert input_placeholder.name in layer.inputs
    assert layer.inputs[input_placeholder.name] is input_placeholder


def test_layer_variables(input_placeholder):
    """
    Test a simple layer determines variables correctly
    """
    layer = SimpleLayer()
    assert isinstance(layer.variables, dict)
    assert len(layer.variables) == 0

    layer(input_placeholder)

    assert len(layer.variables) == 1
    assert layer.weight.name in layer.variables
    assert layer.variables[layer.weight.name] is layer.weight


def test_layer_side_effects(input_placeholder):
    """
    Test a simple layer determines side effects correctly
    """
    layer = SimpleLayer()
    assert isinstance(layer.side_effects, dict)
    assert len(layer.side_effects) == 0

    layer(input_placeholder)

    assert len(layer.side_effects) == 1
    assert isinstance(list(layer.side_effects.values())[0], ng.AssignOp)


def test_layer_outputs(input_placeholder):
    """
    Test a simple layer determines outputs correctly
    """
    layer = SimpleLayer()
    assert isinstance(layer.outputs, dict)
    assert len(layer.outputs) == 0

    out = layer(input_placeholder)

    assert len(layer.outputs) == 1
    assert out.name in layer.outputs
    assert layer.outputs[out.name] is out


def test_nested_layer_scopes(input_placeholder):
    """
    Test nested layers have ops with nested scope names
    """
    layer = NestedLayer()
    layer(input_placeholder)

    outer_scope = layer.name
    inner_scope = layer.inner_layer.name.rsplit("/", 1)[-1]
    for op in layer:
        assert op.scope.name == "/".join([outer_scope, inner_scope])


def test_layer_metadata(input_placeholder):
    """
    Test that layer metadata is added to all created ops
    """
    layer = SimpleLayer()
    layer(input_placeholder)

    for op in layer:
        # Metadata for TensorValueOps is the same as the AssignableTensorOps
        # they read from. This means the TensorValueOp wrapping input_
        # placeholder does not have the layer's metadata.
        if not isinstance(op, ng.TensorValueOp):
            for key, value in layer.metadata.items():
                assert key in op.metadata
                assert op.metadata[key] == value

    assert "label" in layer.variables[layer.name + "/W"].metadata
    assert layer.variables[layer.name + "/W"].metadata["label"] == LABELS["weight"]


def test_scope_ops(input_placeholder):
    """
    Test scope_ops creates a subgraph with correct attributes
    """

    with scope_ops(name="foo") as subgraph:
        w = ng.variable(ng.make_axis(), initial_value=1, name="W")
        y = w * input_placeholder
        z = y + 4
        v1 = ng.persistent_tensor(w.axes, initial_value=0, name="effect1")
        v2 = ng.persistent_tensor(w.axes, initial_value=0, name="effect2")
        ng.sequential([ng.assign(v1, w), ng.assign(v2, w), z]).named("output")

    assert len(subgraph.inputs) == 1
    assert input_placeholder.name in subgraph.inputs

    assert len(subgraph.variables) == 1
    assert "foo/W" in subgraph.variables

    assert len(subgraph.outputs) == 1
    assert "foo/output" in subgraph.outputs

    assert len(subgraph.side_effects) == 2


def test_mode_setting(input_placeholder):
    """
    Modes can be specified when scoping ops. The `modes` attribute should return all ops
    in the subgraph + any on which they depend.
    """
    w = ng.variable(ng.make_axis(), initial_value=1, name="W")
    with scope_ops(name="mode_scope", mode="test") as subgraph:
        w * input_placeholder

    for op in subgraph:
        if not isinstance(op, ng.TensorValueOp):
            assert "mode" in op.metadata
            assert op.metadata["mode"] == "test"

    assert "test" in subgraph.modes
    assert w not in subgraph.ops
    # Make sure all ops in subgraph are in mode
    for op in [w] + subgraph.ops:
        assert op in subgraph.modes["test"].ops


def test_layer_mode_setting(input_placeholder):
    """
    Make sure that Layer.inference_mode sets the correct mode for the graph
    """
    layer = SimpleLayer()
    with Layer.inference_mode_on():
        layer(input_placeholder)

    assert "inference" in layer.modes
    # Make sure all ops in subgraph are in mode
    for op in layer:
        assert op in layer.modes["inference"].ops


def test_dual_scope_layer(input_placeholder):
    """
    If a layer is reused in multiple containers, ops created in each one should have
    distinct nested scopes.
    """

    inner_layer = SimpleLayer()
    container1 = NestedLayer(inner_layer, name="nest1")
    container2 = NestedLayer(inner_layer, name="nest2")

    container1(input_placeholder)
    container2(input_placeholder)

    for op in container1:
        assert op.scope.name == "/".join([container1.name, inner_layer.name])
    for op in container2:
        assert op.scope.name == "/".join([container2.name, inner_layer.name])

    # Variables created in container1 should be reused in container2
    assert list(container1.variables.values())[0] is list(container2.variables.values())[0]


def test_computational_graph_capture(input_placeholder):
    """
    ComputationalGraph object should capture all ops created after it is instantiated.
    """

    cg = ComputationalGraph()
    w = ng.variable(ng.make_axis(), initial_value=1)
    z = w * input_placeholder

    for op in [w, z]:
        assert op in cg.ops

    assert input_placeholder not in cg.ops
    assert w.name in cg.variables


def test_subgraph_scopes_attribute(input_placeholder):
    """
    Subgraphs should have a scopes attribute that contains scope-subgraph pairs.
    """

    subgraph = SubGraph()
    with scope_ops("scope1", subgraph=subgraph):
        w1 = ng.variable(ng.make_axis(), initial_value=1)
        w1 * input_placeholder

    assert len(subgraph.scopes) == 1
    assert "scope1" in subgraph.scopes
    assert len(subgraph.scopes["scope1"].scopes) == 1

    with scope_ops("scope2", subgraph=subgraph):
        w2 = ng.variable(ng.make_axis(), initial_value=1)
        w2 * input_placeholder

    assert len(subgraph.scopes) == 2
    assert "scope2" in subgraph.scopes
    assert len(subgraph.scopes["scope2"].scopes) == 1


def test_scopes_generation(input_placeholder):
    """
    Scopes should be automatically created using layers and can be obtained from the
    ComputationalGraph object.
    """

    cg = ComputationalGraph()

    layer = NestedLayer(SimpleLayer(name="inner"), name="outer")
    layer(input_placeholder)
    assert "outer/inner" in cg.scopes
    for op in layer:
        assert op in cg.scopes["outer/inner"]
