import pytest
import ngraph as ng
from ngraph.frontends.neon import neon_layer, LABELS, Layer


class SimpleLayer(Layer):

    @neon_layer()
    def __call__(self, in_obj):
        if not self.initialized:
            w_axis = ng.make_axis()
            self.weight = ng.variable(axes=[w_axis],
                                      initial_value=2,
                                      metadata={"label": LABELS["weight"]})
            self.side_effect = ng.persistent_tensor(axes=[w_axis],
                                                    initial_value=0)

        return ng.sequential([ng.assign(self.side_effect, self.weight),
                              self.weight * in_obj])


class NestedLayer(Layer):

    def __init__(self):
        super(NestedLayer, self).__init__()
        self.inner_layer = SimpleLayer()

    @neon_layer()
    def __call__(self, in_obj):

        return self.inner_layer(in_obj)


@pytest.fixture(params=[32])
def batch_size(request):
    return request.param


@pytest.fixture(params=[4, 32])
def input_size(request):
    return request.param


def test_layer_inputs(input_placeholder):
    layer = SimpleLayer()
    assert layer.inputs is None

    layer(input_placeholder)

    assert len(layer.inputs) == 1
    assert layer.inputs[0] is input_placeholder


def test_layer_variables(input_placeholder):
    layer = SimpleLayer()
    assert layer.variables is None

    layer(input_placeholder)

    assert len(layer.variables) == 1
    assert layer.variables[0] is layer.weight


def test_layer_side_effects(input_placeholder):
    layer = SimpleLayer()
    assert layer.side_effects is None

    layer(input_placeholder)

    assert len(layer.side_effects) == 1
    assert isinstance(layer.side_effects[0], ng.AssignOp)


def test_layer_metadata(input_placeholder):
    layer = SimpleLayer()
    layer(input_placeholder)

    for op in layer.ops[0]:
        assert "neon_layer" in op.metadata
        assert len(op.metadata["neon_layer"]) == 1
        assert op.metadata["neon_layer"] == ["simplelayer"]

    assert "label" in layer.variables[0].metadata
    assert layer.variables[0].metadata["label"] == LABELS["weight"]


def test_nested_layer_metadata(input_placeholder):
    layer = NestedLayer()
    layer(input_placeholder)

    for op in layer.ops[0]:
        assert "neon_layer" in op.metadata
        assert len(op.metadata["neon_layer"]) == 2
        assert op.metadata["neon_layer"] == ["nestedlayer", "simplelayer"]
