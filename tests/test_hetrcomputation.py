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
import collections
import pytest
from ngraph.testing import ExecutorFactory
from ngraph.util.ordered import OrderedSet
import ngraph as ng
from ngraph.transformers.passes.hetrpasses import DeviceAssignPass, \
    CommunicationPass
from ngraph.transformers.base import transformer_choices
from ngraph.factory.comm_nodes import GatherSendOp, GatherRecvOp, ScatterSendOp, ScatterRecvOp
from ngraph.transformers import (set_transformer_factory,
                                 make_transformer_factory)


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
        expected_transformers = set()

        class MockHetr(object):
            def __init__(self):
                self.transformers = set()

            def register_transformer(self, transformer):
                self.transformers.add(transformer)

        hetr = MockHetr()
        obj = DeviceAssignPass(hetr, default_device, default_device_id)

        obj.do_pass(graph_op, ex.transformer)

        for op in graph_op_metadata.keys():
            assert op.metadata['device'] == graph_op_metadata[op][0]
            assert op.metadata['device_id'] == graph_op_metadata[op][1]
            assert op.metadata['transformer'] == graph_op_metadata[op][0] +  \
                str(graph_op_metadata[op][1])

            expected_transformers.add(op.metadata['transformer'])
        assert hetr.transformers == expected_transformers


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

    # Run the hetr passes one by one, and verify they did the expected things to the graph
    check_device_assign_pass("numpy", "0", graph_op_metadata, graph_ops)
    check_communication_pass(ops_to_transform=graph_ops,
                             expected_recv_nodes=[x_plus_y])


def test_distributed_graph_plus_one(hetr_factory):
    H = ng.make_axis(length=4, name='height')
    W = ng.make_axis(length=6, name='width')
    x = ng.placeholder(axes=[H, W])
    with ng.metadata(device_id=('1', '2'), parallel=W):
        x_plus_one = x + 1

    np_x = np.random.randint(100, size=[H.length, W.length])
    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_one, x)
        res = computation(np_x)
        np.testing.assert_array_equal(res, np_x + 1)


def test_distributed_graph_plus_two(hetr_factory):
    H = ng.make_axis(length=4, name='height')
    W = ng.make_axis(length=6, name='width')
    x = ng.placeholder(axes=[H, W])
    with ng.metadata(device_id=('1', '2'), parallel=W):
        x_plus_one = x + 1
    x_plus_two = x_plus_one + 1

    np_x = np.random.randint(100, size=[H.length, W.length])
    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_two, x)
        res = computation(np_x)
        np.testing.assert_array_equal(res, np_x + 2)


def test_from_device(hetr_factory):
    with ng.metadata(device_id='1'):
        x = ng.placeholder(())
    x_plus_one = x + 1

    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_one, x)
        for i in [10, 20, 30]:
            assert computation(i) == i + 1


def test_to_device(hetr_factory):
    x = ng.placeholder(())
    with ng.metadata(device_id='1'):
        x_plus_one = x + 1

    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_one, x)
        for i in [10, 20, 30]:
            assert computation(i) == i + 1


def test_to_and_from_device(hetr_factory):
    x = ng.placeholder(())
    with ng.metadata(device_id='1'):
        x_plus_one = x + 1
    x_plus_two = x_plus_one + 1

    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_two, x)
        for i in [10, 20, 30]:
            assert computation(i) == i + 2


def test_computation_return_list(hetr_factory):
    with ng.metadata(device_id='1'):
        x = ng.placeholder(())
    x_plus_one = x + 1
    x_plus_two = x + 2
    x_mul_three = x * 3

    with ExecutorFactory() as ex:
        computation = ex.executor([x_plus_one, x_plus_two, x_mul_three], x)
        for i in [10, 20, 30]:
            assert computation(i) == (i + 1, i + 2, i * 3)


def test_gpu_send_and_recv(hetr_factory):
    # skip if gpu unavailable
    if 'gpu' not in transformer_choices():
        pytest.skip("GPUTransformer not available")

    # put x+1 on cpu numpy
    with ng.metadata(device='numpy'):
        x = ng.placeholder(())
        x_plus_one = x + 1
    # put x+2 on gpu numpy
    with ng.metadata(device='gpu'):
        x_plus_two = x_plus_one + 1

    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_two, x)
        for i in [10, 20, 30]:
            assert computation(i) == i + 2

    # put x+1 on gpu numpy
    with ng.metadata(device='gpu'):
        x = ng.placeholder(())
        x_plus_one = x + 1
    # put x+2 on cpu numpy
    with ng.metadata(device='numpy'):
        x_plus_two = x_plus_one + 1

    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_two, x)
        for i in [10, 20, 30]:
            assert computation(i) == i + 2


def test_return_type(hetr_factory):
    x = ng.placeholder(())
    with ExecutorFactory() as ex:
        c0 = ex.executor(x, x)
        c1 = ex.executor([x], x)

        r0 = c0(1)
        assert r0 == 1

        r1 = c1(1)
        assert isinstance(r1, collections.Sequence)
        assert r1[0] == 1


def test_empty_computation(hetr_factory):
    with ExecutorFactory() as ex:
        computation = ex.executor(None)
        res = computation()
        assert not res


def test_wrong_placeholders(hetr_factory):
    x = ng.placeholder(())
    with ExecutorFactory() as ex:
        c = ex.executor(x, x)

        with pytest.raises(AssertionError):
            c()

        with pytest.raises(AssertionError):
            c(1, 2)

        assert c(1) == 1


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
        from_node.metadata['transformer'] = None
        from_node.metadata['parallel'] = t['parallel_axis']
        from_node.metadata['host_transformer'] = None

        to_node = ng.placeholder(axes=t['axes'])
        to_node.metadata['device'] = None
        to_node.metadata['device_id'] = t['device_id']
        to_node.metadata['transformer'] = None
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
