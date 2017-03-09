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

from ngraph.testing import ExecutorFactory
from ngraph.util.ordered import OrderedSet
import ngraph as ng
from ngraph.transformers.passes.hetrpasses import DeviceAssignPass, \
    CommunicationPass, ChildTransformerPass
from ngraph.transformers.base import transformer_choices
from ngraph.factory.comm_nodes import GatherSendOp, GatherRecvOp, ScatterSendOp, ScatterRecvOp
from ngraph.transformers import (set_transformer_factory,
                                 make_transformer_factory)
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
    set_transformer_factory(make_transformer_factory('hetr'))

    with ExecutorFactory() as ex:
        # Build the hetr computation object
        if isinstance(placeholder, tuple):
            computation = ex.executor(ops, *placeholder)
        else:
            computation = ex.executor(ops, placeholder)
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
    set_transformer_factory(make_transformer_factory('hetr'))

    with ExecutorFactory() as ex:
        transformers = set()
        expected_transformers = set()
        obj = DeviceAssignPass(default_device, default_device_id, transformers)

        obj.do_pass(graph_op, ex.transformer)

        for op in graph_op_metadata.keys():
            assert op.metadata['device'] == graph_op_metadata[op][0]
            assert op.metadata['device_id'] == graph_op_metadata[op][1]
            assert op.metadata['transformer'] == graph_op_metadata[op][0] +  \
                str(graph_op_metadata[op][1])

            expected_transformers.add(op.metadata['transformer'])
        assert transformers == expected_transformers


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
    set_transformer_factory(make_transformer_factory('hetr'))
    with ExecutorFactory() as ex:
        send_nodes = OrderedSet()
        obj = CommunicationPass(send_nodes)
        obj.do_pass(ops_to_transform, ex.transformer)

        op_list_instance_type = list()
        num_expected_sendnodes = len(expected_recv_nodes)

        # Count if the communication pass inserted the expected number of send nodes
        assert num_expected_sendnodes == len(send_nodes)

        # verify if Recv nodes are inserted in the right place
        for op in expected_recv_nodes:
            for each_arg in op.args:
                op_list_instance_type.append(type(each_arg))

            if (ng.factory.comm_nodes.NumpyQueueRecvOp in op_list_instance_type or
                ng.factory.comm_nodes.NumpyQueueGatherRecvOp in op_list_instance_type or
                    ng.factory.comm_nodes.NumpyQueueScatterRecvOp in
                    op_list_instance_type) is False:
                assert False
            del op_list_instance_type[:]


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
    set_transformer_factory(make_transformer_factory('hetr'))
    with ExecutorFactory() as ex:
        obj.do_pass(graph_ops, ex.transformer)
    assert set(transformer_list) == set(obj.transformer_list)


def test_distributed_graph_plus_one():

    # Build the graph
    H = ng.make_axis(length=4, name='height')
    W = ng.make_axis(length=6, name='width')

    x = ng.placeholder(axes=[H, W])
    with ng.metadata(device_id=('1', '2'), parallel=W):
        x_plus_one = x + 1

    np_x = np.random.randint(100, size=[H.length, W.length])
    np_result = np.add(np_x, 1)
    check_result_values(input_vector=[np_x],
                        result_expected=[np_result],
                        placeholder=x,
                        ops=OrderedSet([x_plus_one]))


def test_distributed_graph_plus_two():

    # Build the graph
    H = ng.make_axis(length=4, name='height')
    W = ng.make_axis(length=6, name='width')

    x = ng.placeholder(axes=[H, W])
    with ng.metadata(device_id=('1', '2'), parallel=W):
        x_plus_one = x + 1

    x_plus_two = x_plus_one + 1

    np_x = np.random.randint(100, size=[H.length, W.length])
    np_result = np.add(np_x, 2)
    check_result_values(input_vector=[np_x],
                        result_expected=[np_result],
                        placeholder=x,
                        ops=OrderedSet([x_plus_two]))


def test_simple_graph():

    # Build the graph
    with ng.metadata(device_id='1'):
        x = ng.placeholder(())

    x_plus_one = x + 1

    check_result_values(input_vector=[10, 20, 30],
                        result_expected=[(11, 21, 31)],
                        placeholder=x, ops=OrderedSet([x_plus_one]))

    # Build the graph
    x = ng.placeholder(())
    with ng.metadata(device_id='1'):
        x_plus_one = x + 1

    check_result_values(input_vector=[10, 20, 30],
                        result_expected=[(11, 21, 31)],
                        placeholder=x, ops=OrderedSet([x_plus_one]))

    # Build the graph
    x = ng.placeholder(())
    with ng.metadata(device_id='1'):
        x_plus_one = x + 1
    x_plus_two = x_plus_one + 1

    check_result_values(input_vector=[10, 20, 30],
                        result_expected=[(12, 22, 32)],
                        placeholder=x, ops=OrderedSet([x_plus_two]))


def test_multiple_computations():
    # Build the graph
    with ng.metadata(device_id='1'):
        x = ng.placeholder(())

    x_plus_one = x + 1
    x_plus_two = x + 2
    x_mul_three = x * 3

    check_result_values(input_vector=[10, 20, 30],
                        result_expected=[(11, 12, 30),
                                         (21, 22, 60),
                                         (31, 32, 90)],
                        placeholder=x,
                        ops=OrderedSet([x_plus_one, x_plus_two, x_mul_three]))


def test_scatter_gather_graph():
    # Build the graph
    W = ng.make_axis(length=6, name='width')

    with ng.metadata(device_id='0'):
        x = ng.placeholder(())
        z = ng.placeholder(())

    with ng.metadata(device_id=('1', '2'), parallel=W):
        y = ng.placeholder(())

    x_plus_z = x + z  # Does not create a recv node
    x_plus_y = x + y  # creates a gather recv node

    # Build the graph metadata
    graph_ops = OrderedSet([x, y, z, x_plus_z, x_plus_y])

    graph_op_metadata = {op: list() for op in graph_ops}
    graph_op_metadata[x] = ["numpy", '0']
    graph_op_metadata[z] = ["numpy", '0']
    graph_op_metadata[y] = ["numpy", ('1', '2')]
    graph_op_metadata[x_plus_z] = ["numpy", ('0')]
    graph_op_metadata[x_plus_y] = ["numpy", ('0')]

    check_device_assign_pass("numpy", "0", graph_op_metadata, graph_ops)

    check_communication_pass(
        ops_to_transform=graph_ops,
        expected_recv_nodes=[x_plus_y])


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
    set_transformer_factory(make_transformer_factory('hetr'))
    with ExecutorFactory() as ex:
        computation = ex.executor(None)
        res = computation()
        assert not res


def test_scatter_gather_node_axes():
    ax_A = ng.make_axis(64)
    ax_B = ng.make_axis(128)
    ax_C = ng.make_axis(255)

    tests = [
        {
            'axes': ng.make_axes([ax_A]),
            'parallel_axis': ax_A,
            'slices': [[slice(0, 32, 1)], [slice(32, 64, 1)]],
            'device_id': (0, 1)
        },
        {
            'axes': ng.make_axes([ax_A, ax_B]),
            'parallel_axis': ax_A,
            'slices': [[slice(0, 21, 1), slice(None)],
                       [slice(21, 42, 1), slice(None)],
                       [slice(42, 64, 1), slice(None)]],
            'device_id': (0, 1, 2)
        },
        {
            'axes': ng.make_axes([ax_A, ax_B, ax_C]),
            'parallel_axis': ax_A,
            'slices': [[slice(0, 12, 1), slice(None), slice(None)],
                       [slice(12, 24, 1), slice(None), slice(None)],
                       [slice(24, 36, 1), slice(None), slice(None)],
                       [slice(36, 48, 1), slice(None), slice(None)],
                       [slice(48, 64, 1), slice(None), slice(None)]],
            'device_id': (0, 1, 2, 3, 4)
        },
        {
            'axes': ng.make_axes([ax_A, ax_B, ax_C]),
            'parallel_axis': ax_C,
            'slices': [[slice(None), slice(None), slice(0, 127, 1)],
                       [slice(None), slice(None), slice(127, 255, 1)]],
            'device_id': (0, 1)
        }
    ]

    for t in tests:
        from_node = ng.placeholder(axes=t['axes'])
        from_node.metadata['device'] = None
        from_node.metadata['device_id'] = t['device_id']
        from_node.metadata['parallel'] = t['parallel_axis']
        from_node.metadata['host_transformer'] = None

        to_node = ng.placeholder(axes=t['axes'])
        to_node.metadata['device'] = None
        to_node.metadata['device_id'] = t['device_id']
        to_node.metadata['parallel'] = t['parallel_axis']
        to_node.metadata['host_transformer'] = None

        gather_send_op = GatherSendOp(from_node=from_node)
        assert t['axes'] == gather_send_op.axes

        gather_recv_op = GatherRecvOp(from_node=from_node,
                                      to_node=to_node,
                                      send_node=gather_send_op)
        assert t['axes'] == gather_recv_op.axes
        assert t['slices'] == gather_recv_op.slices

        scatter_send_op = ScatterSendOp(from_node=from_node,
                                        to_node=to_node)
        assert t['axes'] == scatter_send_op.axes
        assert t['slices'] == scatter_send_op.slices

        scatter_recv_op = ScatterRecvOp(to_node=to_node,
                                        send_node=scatter_send_op)
        assert t['axes'] == scatter_recv_op.axes
