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
import pytest
from ngraph.testing import ExecutorFactory
from ngraph.util.ordered import OrderedSet
import ngraph as ng
from ngraph.transformers.passes.hetrpasses import DeviceAssignPass, \
    CommunicationPass
from ngraph.transformers.base import transformer_choices

pytestmark = pytest.mark.hetr_only("module")


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


def test_hetr_graph_passes(transformer_factory):

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


def test_distributed_graph_plus_one(transformer_factory):
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


def test_distributed_graph_plus_two(transformer_factory):
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


def test_from_device(transformer_factory):
    with ng.metadata(device_id='1'):
        x = ng.placeholder(())
    x_plus_one = x + 1

    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_one, x)
        for i in [10, 20, 30]:
            assert computation(i) == i + 1


def test_to_device(transformer_factory):
    x = ng.placeholder(())
    with ng.metadata(device_id='1'):
        x_plus_one = x + 1

    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_one, x)
        for i in [10, 20, 30]:
            assert computation(i) == i + 1


def test_to_and_from_device(transformer_factory):
    x = ng.placeholder(())
    with ng.metadata(device_id='1'):
        x_plus_one = x + 1
    x_plus_two = x_plus_one + 1

    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_two, x)
        for i in [10, 20, 30]:
            assert computation(i) == i + 2


def test_computation_return_list(transformer_factory):
    with ng.metadata(device_id='1'):
        x = ng.placeholder(())
    x_plus_one = x + 1
    x_plus_two = x + 2
    x_mul_three = x * 3

    with ExecutorFactory() as ex:
        computation = ex.executor([x_plus_one, x_plus_two, x_mul_three], x)
        for i in [10, 20, 30]:
            assert computation(i) == (i + 1, i + 2, i * 3)


@pytest.mark.hetr_gpu_only
def test_gpu_send_and_recv(transformer_factory):
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
