# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import pytest
import six
from six import BytesIO
from google.protobuf.json_format import Parse

import numpy as np
import ngraph as ng
from ngraph.op_graph.serde import ops_pb2
from ngraph.op_graph.serde import serde_weights
from ngraph.op_graph.serde import serde
from ngraph.testing import executor, ExecutorFactory


def assign_ops(ops, values):
    assign_ops = [ng.AssignOp(op, value) for op, value in zip(ops, values)]
    return ng.sequential(assign_ops)


def test_serialize_and_deserialize_single_raw_np():
    x = np.random.random((1, ))

    # write out values in x
    f = BytesIO()
    serde_weights.write_raw_np(x, f)

    # reset file so it appears to be freshly opened for deserialize_weights
    f.seek(0)
    de_x = f.read()

    assert de_x == x.tostring()


def test_serialize_and_deserialize_multi_np():
    x = np.random.random((1, ))
    y = np.random.random((2, 3))
    z = np.random.random((1, 5, 2))
    values = {'x': x, 'y': y, 'z': z}

    # write out values in x
    f = BytesIO()
    serde_weights.write_np_values(values, f)

    # reset file so it appears to be freshly opened for deserialize_weights
    f.seek(0)
    de_values = serde_weights.read_np_values(f)

    assert values.keys() == de_values.keys()
    for k, v in values.items():
        assert (de_values[k] == v).all()


@pytest.config.cpu_enabled_only(reason="Only CPU supports dynamic graph changes")
@pytest.mark.transformer_dependent
def test_extract_op(transformer_factory):
    # set up an op and Assign a value to it so we can read it out
    axes = ng.make_axes([
        ng.make_axis(name='A', length=2),
        ng.make_axis(name='B', length=3),
    ])
    x_op = ng.variable(axes)
    assign_op = ng.AssignOp(x_op, 1)

    # extract values out of it and make sure they match expected results
    with executor(assign_op) as comp_assignment:
        t = comp_assignment.transformer
        comp_assignment()
        x_out = serde_weights.extract_op(t, x_op)
    assert (x_out == np.ones(axes.lengths)).all()


@pytest.config.cpu_enabled_only(reason="Only CPU supports dynamic graph changes")
@pytest.mark.transformer_dependent
def test_extract_many_ops(transformer_factory):
    """
    Create NUM_OPS, fill them with 0, 1, 2, ... then check that
    serde_weights.extract_ops is able to extract the correct uuid/value pairs.
    """
    NUM_OPS = 3

    # set up an op and Assign a value to it so we can read it out
    axes = ng.make_axes([
        ng.make_axis(name='A', length=2),
        ng.make_axis(name='B', length=3),
    ])
    variable_ops = [ng.variable(axes) for _ in range(NUM_OPS)]
    assign_sequential = assign_ops(variable_ops, range(NUM_OPS))

    with executor(assign_sequential) as assign_computation:
        t = assign_computation.transformer
        # Set values manually
        assign_computation()

        # extract values out of it and make sure they match expected results
        weights = serde_weights.extract_ops(t, variable_ops)

    for i, variable_op in enumerate(variable_ops):
        np.testing.assert_allclose(weights[variable_op.uuid.bytes], i)


@pytest.config.cpu_enabled_only(reason="Only CPU supports dynamic graph changes")
@pytest.mark.transformer_dependent
def test_set_op_value(transformer_factory):
    """
    set up a variable, then use serde_weights.set_op_value to inject a value
    into the graph.  Then double check that the value was injected.
    """
    axes = ng.make_axes([
        ng.make_axis(name='A', length=2),
        ng.make_axis(name='B', length=3),
    ])
    x_op = ng.variable(axes)

    with ExecutorFactory() as ex:
        t = ex.transformer
        value = np.ones(axes.lengths)
        serde_weights.set_op_value(t, x_op, value)

        # extract values out of it and make sure they match expected results
        x_out = serde_weights.extract_op(t, x_op)
    assert (x_out == value).all()


@pytest.config.cpu_enabled_only(reason="Only CPU supports dynamic graph changes")
@pytest.mark.transformer_dependent
def test_set_op_values(transformer_factory):
    NUM_OPS = 3

    # set up an op and Assign a value to it so we can read it out
    axes = ng.make_axes([
        ng.make_axis(name='A', length=2),
        ng.make_axis(name='B', length=3),
    ])
    variable_ops = [ng.variable(axes) for i in range(NUM_OPS)]
    values = [np.ones(axes.lengths) * i for i in range(NUM_OPS)]
    with ExecutorFactory() as ex:
        t = ex.transformer
        serde_weights.set_op_values(
            t,
            variable_ops, {
                op.uuid.bytes: value
                for op, value in zip(variable_ops, values)
            }
        )
        # extract values out of it and make sure they match expected results
        funcs = [t.computation(op) for op in variable_ops]
        op_values = [func() for func in funcs]

    np.testing.assert_allclose(values, op_values)


def make_axes(lengths):
    return ng.make_axes([ng.make_axis(length) for length in lengths])


def test_json_dumps_manifest():
    # create a list of variables with varying lengths and dtype
    x = np.zeros((2, 3))

    js = serde_weights.json_dumps_manifest({'x': x})

    # now deserialize js formatted manifest and make sure it has the right data in it
    manifest = ops_pb2.TensorManifest()
    Parse(js, manifest)

    assert len(manifest.pairs) == 1
    t = manifest.pairs[0]
    assert t.uuid.uuid == six.b('x')
    assert t.info.dtype == ops_pb2.FLOAT64
    assert t.info.shape == [2, 3]


@pytest.config.cpu_enabled_only(reason="Only CPU supports dynamic graph changes")
@pytest.mark.transformer_dependent
def test_round_trip(transformer_factory):
    # set up an op and Assign a value to it so we can read it out
    axes = ng.make_axes([
        ng.make_axis(name='A', length=2),
        ng.make_axis(name='B', length=3),
    ])
    x_op = ng.variable(axes)

    assign_op = ng.AssignOp(x_op, 1)

    with executor(assign_op) as assign_computation:
        t = assign_computation.transformer

        # Set initial value
        assign_computation()

        # Test value
        np.testing.assert_allclose(serde_weights.extract_op(t, x_op), 1)

        # write out values in x and graph
        f = BytesIO()

        # ## EXAMPLE OF HOW TO FULLY SERIALIZE A GRAPH ###
        serde_weights.serialize_weights(t, [x_op], f)
        graph_string = serde.serialize_graph([x_op])
        # ## /EXAMPLE OF HOW TO FULLY SERIALIZE A GRAPH ###

        f.seek(0)

        # ## EXAMPLE OF HOW TO FULLY DESERIALIZE A GRAPH ###
        new_ops = serde.deserialize_graph(graph_string)
        serde_weights.deserialize_weights(t, new_ops, f)
        # ## /EXAMPLE OF HOW TO FULLY DESERIALIZE A GRAPH ###

        np.testing.assert_allclose(serde_weights.extract_op(t, new_ops[0]), 1)
