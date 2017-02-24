# ----------------------------------------------------------------------------
# copyright 2016 Nervana Systems Inc.
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
import numpy as np

from ngraph.util.ordered import OrderedSet
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.transformers.passes.hetrpasses import DeviceAssignPass, \
    CommunicationPass, ChildTransformerPass
from ngraph.transformers.base import transformer_choices
from ngraph.op_graph.communication import Gather_Send, Gather_Recv, Scatter_Send, Scatter_Recv
import pytest


def check_result_values(input_vector, result_expected, placeholder, ops=OrderedSet(), *args):
    """
    This function checks the result values return by the hetr computation object
    against the expected result values
    it also checks if the value returned by the hetr object matches the order in
    the expected result list

    :param: input_vector: list specifying the differnt values to be passed to
            the placeholder
    :param: result_expected: list of tuples specifying the expected result
            values from the hetr computation object
    :param: placeholder: list of placeholder to be passed for hetrcomputation
    :param: ops: list of result handlers to be paased for hetrcomputation

    """
    # Select the transformer
    transformer = ngt.make_transformer_factory('hetr')()

    # Build the hetr computation object
    if isinstance(placeholder, tuple):
        computation = transformer.computation(ops, *placeholder)
    else:
        computation = transformer.computation(ops, placeholder)
    result_obtained = []

    # Check for the return result list
    for i in input_vector:
        if isinstance(i, tuple):
            result_obtained.append(computation(*i))
        else:
            result_obtained.append(computation(i))

    # if return result is tuple
    if len(result_expected) > 1:
        np.testing.assert_array_equal(result_expected, result_obtained)

    # if return result is  scalar
    else:
        assert (np.array(tuple(result_obtained)) ==
                np.array(result_expected[0])).all()

    transformer.close()


def check_device_assign_pass(default_device, default_device_id,
                             graph_op_metadata, graph_op=OrderedSet(), *args):
    """
    The Device assign pass should inject the metadata{device_id, device} as
    specified by the user for each op,
    if not specified then the default {device_id:0, device:numpy} should be
    inserted for each op.

    :param: default_device: string, the default device for each op,
            if not specified by user ex: "numpy"
    :param: default_device_id: string, the default device number for each op,
            if not specified by user ex: "0"
    :param: graph_op_metadata: dict, dictionary of list specifying  the expected
            metadata {device_id, device} for each op
    :param: graph_op: list of ops to do the graph traversal

    """
    transformer = ngt.make_transformer_factory('hetr')()

    transformers = set()
    expected_transformers = set()
    obj = DeviceAssignPass(default_device, default_device_id, transformers)

    obj.do_pass(graph_op, transformer)

    for op in graph_op_metadata.keys():
        assert op.metadata['device'] == graph_op_metadata[op][0]
        assert op.metadata['device_id'] == graph_op_metadata[op][1]
        assert op.metadata['transformer'] == graph_op_metadata[op][0] +  \
            str(graph_op_metadata[op][1])

        expected_transformers.add(op.metadata['transformer'])
    assert transformers == expected_transformers

    transformer.close()


def check_communication_pass(ops_to_transform, expected_recv_nodes):
    """
    The communication pass should insert send/recv nodes wherever
    the metadata[transformer] differs between nodes.
    This checks that the recv nodes are inserted in the right place, and counts
    that the expected number of send
    nodes are found.

    :param ops_to_transform: list of ops to do the garph traversal
    :param expected_recv_nodes: lits of ops where receive nodes are expected to
           be inserted after the communication pass

    """
    transformer = ngt.make_transformer_factory('hetr')()

    send_nodes = OrderedSet()
    scatter_shared_queues = list()
    gather_shared_queues = list()
    obj = CommunicationPass(send_nodes, scatter_shared_queues, gather_shared_queues)
    obj.do_pass(ops_to_transform, transformer)

    op_list_instance_type = list()
    num_expected_sendnodes = len(expected_recv_nodes)

    # Count if the communication pass inserted the expected number of send nodes
    assert num_expected_sendnodes == len(send_nodes)

    # verify if Recv nodes are inserted in the right place
    for op in expected_recv_nodes:
        for each_arg in op.args:
            op_list_instance_type.append(type(each_arg))

        if (ng.op_graph.communication.Recv in op_list_instance_type or
            ng.op_graph.communication.Gather_Recv in op_list_instance_type or
                ng.op_graph.communication.Scatter_Recv in op_list_instance_type) is False:
            assert False
        del op_list_instance_type[:]

    transformer.close()


def test_hetr_graph_passes():

    # Build the graph
    with ng.metadata(device_id='1'):
        x = ng.placeholder(())

    y = ng.placeholder(())
    x_plus_y = x + y

    # Build the graph metadata
    graph_ops = OrderedSet([x_plus_y, x, y])

    graph_op_metadata = {op: list() for op in graph_ops}
    graph_op_metadata[x] = ["numpy", '1']
    graph_op_metadata[y] = ["numpy", '0']
    graph_op_metadata[x_plus_y] = ["numpy", '0']

    transformer_list = ["numpy1", "numpy0"]

    # Run the hetr passes one by one, and verify they did the expected things to the graph
    check_device_assign_pass("numpy", "0", graph_op_metadata, graph_ops)
    check_communication_pass(ops_to_transform=graph_ops,
                             expected_recv_nodes=[x_plus_y])

    # Check if the hetr pass (childTransfromer pass) generates the expected transformer list
    obj = ChildTransformerPass([])
    transformer = ngt.make_transformer_factory('hetr')()
    obj.do_pass(graph_ops, transformer)
    transformer.close()
    assert set(transformer_list) == set(obj.transformer_list)


def test_distributed_graph():

    # Build the graph
    H = ng.make_axis(length=4, name='height')
    W = ng.make_axis(length=6, name='width')

    x = ng.placeholder(axes=[H, W])
    y = ng.placeholder(())
    z = ng.placeholder(())
    with ng.metadata(device_id=('1', '2'), parallel=W):
        x_plus_y = x + y

    x_plus_y_plus_z = x_plus_y + z

#    # Build the graph metadata
#    graph_ops = OrderedSet([x_plus_y_plus_z, x_plus_y, x, y, z])
#
#    graph_op_metadata = {op: list() for op in graph_ops}
#    graph_op_metadata[x] = ["numpy", '0']
#    graph_op_metadata[y] = ["numpy", '0']
#    graph_op_metadata[z] = ["numpy", '0']
#    graph_op_metadata[x_plus_y] = ["numpy", ('1', '2')]
#    graph_op_metadata[x_plus_y_plus_z] = ["numpy", '0']
#
#    transformer_list = ["numpy2", "numpy1", "numpy0"]
#
#    # Run the hetr passes one by one, and verify they did the expected things to the graph
#    check_device_assign_pass("numpy", "0", graph_op_metadata, graph_ops)
#    check_communication_pass(
#        ops_to_transform=graph_ops,
#        expected_recv_nodes=[
#            x_plus_y,
#            x_plus_y,
#            x_plus_y_plus_z])
#
#    # Check if the hetr pass (childTransfromer pass) generates the expected transformer list
#    obj = ChildTransformerPass([])
#    transformer = ngt.make_transformer_factory('hetr')()
#    obj.do_pass(graph_ops, transformer)
#    transformer.close()
#
#    assert set(transformer_list) == set(obj.transformer_list)
    pytest.xfail("Some problems due to latest changes from master, fixes in later PR")
    check_result_values(input_vector=[(10, 20, 30), (1, 2, 3)],
                        result_expected=[(60,),
                                         (6,)],
                        placeholder=(x, y, z),
                        ops=OrderedSet([x_plus_y_plus_z]))


def test_simple_graph():

    # Build the graph
    with ng.metadata(device_id='1'):
        x = ng.placeholder(())

    x_plus_one = x + 1

    check_result_values(input_vector=[10, 20, 30],
                        result_expected=[(11, 21, 31)],
                        placeholder=x, ops=OrderedSet([x_plus_one]))

    x_plus_one = x + 1
    x_plus_two = x + 2
    x_mul_three = x * 3

    check_result_values(input_vector=[10, 20, 30],
                        result_expected=[(11, 12, 30),
                                         (21, 22, 60),
                                         (31, 32, 90)],
                        placeholder=x,
                        ops=OrderedSet([x_plus_one, x_plus_two, x_mul_three]))


def test_gpu_send_and_recv():
    # First check whether do we have gputransformer available, if not, xfail
    if 'gpu' not in transformer_choices():
        pytest.skip("GPUTransformer not available")

    # put x+1 on cpu numpy
    with ng.metadata(device='numpy'):
        x = ng.placeholder(())
        x_plus_one = x + 1
    # put x+2 on gpu numpy
    with ng.metadata(device='gpu'):
        x_plus_two = x_plus_one + 1

    check_result_values(input_vector=[10, 20, 30],
                        result_expected=[(12), (22), (32)],
                        placeholder=x, ops=OrderedSet([x_plus_two]))

    # put x+1 on gpu numpy
    with ng.metadata(device='gpu'):
        x = ng.placeholder(())
        x_plus_one = x + 1
    # put x+2 on cpu numpy
    with ng.metadata(device='numpy'):
        x_plus_two = x_plus_one + 1

    check_result_values(input_vector=[10, 20, 30],
                        result_expected=[(12), (22), (32)],
                        placeholder=x, ops=OrderedSet([x_plus_two]))


def test_empty_computation():
    transformer = ngt.make_transformer_factory('hetr')()
    computation = transformer.computation(None)
    res = computation()
    assert not res
    transformer.close()


def test_scatter_gather_node_axes():
    ax_A = ng.make_axis(64)
    ax_B = ng.make_axis(128)
    ax_C = ng.make_axis(255)

    tests = [
        {
            'axes': ng.make_axes([ax_A]),
            'parallel_axis': ax_A,
            'slices': [[slice(0, 32, 1)], [slice(32, 64, 1)]],
            'devices': (0, 1)
        },
        {
            'axes': ng.make_axes([ax_A, ax_B]),
            'parallel_axis': ax_A,
            'slices': [[slice(0, 21, 1), slice(None)],
                       [slice(21, 42, 1), slice(None)],
                       [slice(42, 64, 1), slice(None)]],
            'devices': (0, 1, 2)
        },
        {
            'axes': ng.make_axes([ax_A, ax_B, ax_C]),
            'parallel_axis': ax_A,
            'slices': [[slice(0, 12, 1), slice(None), slice(None)],
                       [slice(12, 24, 1), slice(None), slice(None)],
                       [slice(24, 36, 1), slice(None), slice(None)],
                       [slice(36, 48, 1), slice(None), slice(None)],
                       [slice(48, 64, 1), slice(None), slice(None)]],
            'devices': (0, 1, 2, 3, 4)
        },
        {
            'axes': ng.make_axes([ax_A, ax_B, ax_C]),
            'parallel_axis': ax_C,
            'slices': [[slice(None), slice(None), slice(0, 127, 1)],
                       [slice(None), slice(None), slice(127, 255, 1)]],
            'devices': (0, 1)
        }
    ]

    for t in tests:
        gather_send_node = Gather_Send(from_node=ng.placeholder(()),
                                       axes=t['axes'], queue=None,
                                       device=None, device_id=None)
        assert t['axes'] == gather_send_node.axes

        gather_recv_node = Gather_Recv(axes=t['axes'], dtype=np.float32,
                                       parallel_axis=t['parallel_axis'],
                                       queues=None, send_node=gather_send_node,
                                       device=None, device_id=None,
                                       from_id=t['devices'])
        assert t['axes'] == gather_recv_node.axes
        assert t['slices'] == gather_recv_node.slices

        scatter_send_node = Scatter_Send(from_node=ng.placeholder(()),
                                         axes=t['axes'], parallel_axis=t['parallel_axis'],
                                         queues=None, device=None, device_id=None,
                                         to_id=t['devices'])
        assert t['axes'] == scatter_send_node.axes
        assert t['slices'] == scatter_send_node.slices

        scatter_recv_node = Scatter_Recv(axes=t['axes'], dtype=np.float32,
                                         queue=None, send_node=scatter_send_node,
                                         device=None, device_id=None)
        assert t['axes'] == scatter_recv_node.axes
