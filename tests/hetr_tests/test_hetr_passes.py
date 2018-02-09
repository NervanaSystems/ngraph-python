# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import pytest
from ngraph.testing import ExecutorFactory
from orderedset import OrderedSet
import ngraph as ng
from ngraph.transformers.passes.hetrpasses import DeviceAssignPass, \
    CommunicationPass


pytestmark = pytest.mark.hetr_only


def check_device_assign_pass(default_device, default_device_id,
                             graph_op_metadata, graph_op=OrderedSet(), *args):
    """
    The Device assign pass should inject the metadata{device_id, device} as
    specified by the user for each op,
    if not specified then the default {device_id:0, device:cpu} should be
    inserted for each op.

    :param: default_device: string, the default device for each op,
            if not specified by user ex: "cpu"
    :param: default_device_id: string, the default device number for each op,
            if not specified by user ex: "0"
    :param: graph_op_metadata: dict, dictionary of list specifying  the expected
            metadata {device_id, device} for each op
    :param: graph_op: list of ops to do the graph traversal
    """
    with ExecutorFactory():
        expected_transformers = set()

        class MockHetr(object):

            def __init__(self):
                self.transformers = set()

            def register_transformer(self, transformer):
                self.transformers.add(transformer)

        hetr = MockHetr()
        obj = DeviceAssignPass(hetr, default_device, default_device_id)

        obj.do_pass(ops=graph_op)

        for op in graph_op_metadata.keys():
            assert op.metadata['device'] == graph_op_metadata[op][0]
            assert op.metadata['device_id'] == graph_op_metadata[op][1]
            if isinstance(graph_op_metadata[op][1], (list, tuple)):
                transformer = [graph_op_metadata[op][0] + str(i) for i in graph_op_metadata[op][1]]
            else:
                transformer = graph_op_metadata[op][0] + str(graph_op_metadata[op][1][0])
            assert op.metadata['transformer'] == transformer

            for device_id in graph_op_metadata[op][1]:
                expected_transformers.add(graph_op_metadata[op][0] + device_id)
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
    with ExecutorFactory():
        send_nodes = OrderedSet()
        obj = CommunicationPass(send_nodes)
        obj.do_pass(ops_to_transform)

        op_list_instance_type = list()
        num_expected_sendnodes = len(expected_recv_nodes)

        # Count if the communication pass inserted the expected number of send nodes
        assert num_expected_sendnodes == len(send_nodes)

        # verify if Recv nodes are inserted in the right place
        for op in expected_recv_nodes:
            for each_arg in op.args:
                op_list_instance_type.append(type(each_arg))

            if (ng.op_graph.comm_nodes.CPUMlslRecvOp in op_list_instance_type or
                ng.op_graph.comm_nodes.CPUMlslGatherRecvOp in op_list_instance_type or
                ng.op_graph.comm_nodes.CPUMlslScatterRecvOp in
                    op_list_instance_type or
                ng.op_graph.comm_nodes.GPUQueueRecvOp in op_list_instance_type or
                ng.op_graph.comm_nodes.GPUCudaGatherRecvOp in op_list_instance_type or
                ng.op_graph.comm_nodes.GPUCudaScatterRecvOp in
                    op_list_instance_type) is False:
                assert False
            del op_list_instance_type[:]


def test_singleton_device_id(hetr_device):
    with ng.metadata(device_id=(['1'])):
        x = ng.placeholder(())
    graph_ops = OrderedSet([x])

    graph_op_metadata = {op: list() for op in graph_ops}
    graph_op_metadata[x] = [hetr_device, '1']

    check_device_assign_pass(hetr_device, "0", graph_op_metadata, graph_ops)


def test_scatter_gather_graph(hetr_device):
    # Build the graph
    W = ng.make_axis(length=6, name='width')

    with ng.metadata(device=hetr_device, device_id='0'):
        x = ng.placeholder(())
        z = ng.placeholder(())

    with ng.metadata(device=hetr_device, device_id=('0', '1'), parallel=W):
        y = ng.placeholder(())

    x_plus_z = x + z  # Does not create a recv node
    x_plus_y = x + y  # creates a gather recv node

    # Build the graph metadata
    graph_ops = OrderedSet([x, y, z, x_plus_z, x_plus_y])

    graph_op_metadata = {op: list() for op in graph_ops}
    graph_op_metadata[x] = [hetr_device, '0']
    graph_op_metadata[z] = [hetr_device, '0']
    graph_op_metadata[y] = [hetr_device, ('0', '1')]
    graph_op_metadata[x_plus_z] = [hetr_device, '0']
    graph_op_metadata[x_plus_y] = [hetr_device, '0']

    check_device_assign_pass(hetr_device, "0", graph_op_metadata, graph_ops)

    check_communication_pass(
        ops_to_transform=graph_ops,
        expected_recv_nodes=[x_plus_y])


def test_hetr_graph_passes():

    # Build the graph
    with ng.metadata(device_id='1'):
        x = ng.placeholder(())

    y = ng.placeholder(())
    x_plus_y = x + y

    # Build the graph metadata
    graph_ops = OrderedSet([x_plus_y, x, y])

    graph_op_metadata = {op: list() for op in graph_ops}
    graph_op_metadata[x] = ["cpu", '1']
    graph_op_metadata[y] = ["cpu", '0']
    graph_op_metadata[x_plus_y] = ["cpu", '0']

    # Run the hetr passes one by one, and verify they did the expected things to the graph
    check_device_assign_pass("cpu", "0", graph_op_metadata, graph_ops)
    check_communication_pass(ops_to_transform=graph_ops,
                             expected_recv_nodes=[x_plus_y])
