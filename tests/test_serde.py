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
import os
from copy import deepcopy

import numpy as np
import ngraph as ng
from ngraph.op_graph.op_graph import Op
import ngraph.op_graph.serde.serde as ser
from ngraph.op_graph.serde.serde_pass import SerializationPass
from ngraph.testing.hetr_utils import create_send_recv_graph


def get_simple_graph():
    ax = ng.make_axes([ng.make_axis(name='C', length=1)])
    base_op = ng.constant(5.0, ax)
    simple_graph = ng.log(ng.exp(base_op))
    return base_op, simple_graph


def strip_dict(d):
    """
    For equality testing we need to remove attributes of dicts that are either unique to each
    instance or need more complex equality handling
    """
    keys = ('_NameableValue__name', '_axes', '_args', 'valfun', 'dtype',
            'scale', '_tensor', '_send_node')
    for key in keys:
        if key in d:
            del d[key]


def assert_object_equality(obj1, obj2):
    if hasattr(obj1, '_args'):
        for arg1, arg2 in zip(obj1._args, obj2._args):
            assert_object_equality(arg1, arg2)
    d1 = deepcopy(obj1.__dict__)
    strip_dict(d1)
    d2 = deepcopy(obj2.__dict__)
    strip_dict(d2)
    assert d1 == d2


def test_flattenedaxis_serialization():
    # We do a round robin serialization run with an axis and make sure that they match
    c = ng.make_axis(name='C', length=2)
    h = ng.make_axis(name='H', length=3)
    orig_axis = ng.make_axes([c, h]).flatten()

    pb_axis = ser.axis_to_protobuf(orig_axis)
    py_axis = ser.pb_to_axis(pb_axis)
    assert py_axis.length == orig_axis.length
    # NameableValue name counter is different
    # assert orig_axis.name == py_axis.name
    assert type(py_axis) == type(orig_axis)
    assert orig_axis == py_axis


def test_axis_serialization():
    # We do a round robin serialization run with an axis and make sure that they match
    axis = ng.make_axis(name='C', length=2)
    pb_axis = ser.axis_to_protobuf(axis)
    py_axis = ser.pb_to_axis(pb_axis)
    assert axis.length == py_axis.length
    assert axis.name == py_axis.name
    assert axis == py_axis


def test_tensor_to_protobuf():
    orig_tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    pb_tensor = ser.tensor_to_protobuf(orig_tensor)
    py_tensor = ser.pb_to_tensor(pb_tensor)
    np.testing.assert_allclose(orig_tensor, py_tensor)


def test_scalar_to_protobuf():
    orig_tensor = np.float32(12)
    pb_tensor = ser.tensor_to_protobuf(orig_tensor)
    py_tensor = ser.pb_to_tensor(pb_tensor)
    np.testing.assert_allclose(orig_tensor, py_tensor)


def test_op_to_protobuf():
    axis = ng.make_axis(name='C', length=2)
    axes = ng.make_axes([axis])
    orig_op = ng.placeholder(axes)

    # Test attributes
    orig_op.test0 = 'stringval_attr'
    orig_op.test1 = [-1.0, 4]
    orig_op.test2 = dict(foo=2, you='bar')
    orig_op.test3 = dict()
    orig_op.test4 = slice(1, 3, 5)
    orig_op.test5 = slice(1, 3)
    orig_op.test6 = slice(1, None, 3)
    orig_op.test7 = axis
    orig_op.test8 = axes

    # Test metadata
    orig_op.metadata['test0'] = 'stringval'
    orig_op.metadata['test1'] = [1, 4.0]
    orig_op.metadata['test2'] = dict(hey=1, you=4.0)
    orig_op.metadata['test4'] = dict()
    orig_op.metadata['test5'] = slice(1, 3, 5)
    orig_op.metadata['test6'] = slice(1, 3)
    orig_op.metadata['test7'] = slice(1, None, 5)
    orig_op.metadata['test8'] = axis
    orig_op.metadata['test9'] = axes

    pb_op = ser.op_to_protobuf(orig_op)
    py_op = ser.protobuf_to_op(pb_op)
    assert_object_equality(py_op, orig_op)


def test_op_references():
    # test op references in arbitrary attributes
    orig_op = ng.placeholder(())
    other_op = ng.placeholder(()).named("foo")
    orig_op.op_ref = other_op
    orig_op.many_op_refs = [other_op]
    ser_string = ser.serialize_graph([orig_op], only_return_handle_ops=True)
    py_op = ser.deserialize_graph(ser_string)[0]
    assert py_op.op_ref.name.startswith('foo')
    assert py_op.many_op_refs[0].name.startswith('foo')


def test_full_graph_serialization_endtoend():
    base_op, simple_graph = get_simple_graph()

    ser_string = ser.serialize_graph([simple_graph])
    py_graph = ser.deserialize_graph(ser_string)
    orig_graph = Op.all_op_references([simple_graph])

    # This is actually overkill since the checks of the leaf nodes will recursively
    # check equality up the graph, but we also want to make sure the full set of nodes
    # returned is equal
    for o1, o2 in zip(sorted(py_graph, key=lambda x: x.uuid),
                      sorted(orig_graph, key=lambda x: x.uuid)):
        assert_object_equality(o1, o2)


def test_op_handle_selection():
    """
    When serializing graphs, we can optionally add metadata to
    those nodes we pass in, and return only those nodes when deserializing.

    This is useful for ngraph transparent testing since it is common in
    ngraph to use the final op as the 'handle' to the entire graph.
    """
    base_op, simple_graph = get_simple_graph()
    ser_string = ser.serialize_graph([simple_graph], only_return_handle_ops=True)
    py_graph = ser.deserialize_graph(ser_string)
    assert len(py_graph) == 1
    assert_object_equality(simple_graph, py_graph[0])


def test_ser_pass():
    _, graph = get_simple_graph()
    ser_pass = SerializationPass('mypass_token')
    fname = ser_pass.tmpfile.name
    ser_pass.do_pass(ops=[graph])
    assert os.path.getsize(fname) > 0
    os.unlink(fname)


def test_hetr_send_recv_graph_serialization():
    """
    test serializing send/recv ops defined in comm_nodes for hetr communication
    """
    z, recv_x, recv_x_plus_one, send_x, x_plus_one, from_node, send_x_plus_one = \
        create_send_recv_graph()
    ser_string = ser.serialize_graph([z])
    py_graph = ser.deserialize_graph(ser_string)
    orig_graph = Op.all_op_references([z])

    for o1, o2 in zip(sorted(py_graph, key=lambda x: x.uuid),
                      sorted(orig_graph, key=lambda x: x.uuid)):
        assert_object_equality(o1, o2)


def test_all_op_references():
    base_op, simple_graph = get_simple_graph()

    leaf_all_ops = Op.all_op_references([simple_graph])
    assert base_op in leaf_all_ops
    assert simple_graph in leaf_all_ops
    base_all_ops = Op.all_op_references([base_op])
    assert base_op in base_all_ops
    assert simple_graph not in base_all_ops
