# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
from ngraph.transformers.passes.passes import GraphBuildingPass
from ngraph.op_graph.comm_nodes import calculate_new_axes
from ngraph.factory.comm_node_factory import get_node_type, CommNodePair
from ngraph.op_graph.op_graph import Op, TensorValueOp
from ngraph.util.hetr_utils import clone_graph
from orderedset import OrderedSet
import socket


class DeviceAssignPass(GraphBuildingPass):
    def __init__(self, hetr, default_device, default_device_id):
        super(DeviceAssignPass, self).__init__()
        self.hetr = hetr
        self.default_device = default_device
        self.default_device_id = default_device_id

    def visit(self, op):
        device = op.metadata.setdefault('device', self.default_device)
        device_id = op.metadata.setdefault('device_id', self.default_device_id)
        transformer = "{}{}".format(device, device_id)
        host_transformer = socket.gethostname()
        op.metadata['host_transformer'] = host_transformer
        if isinstance(op.metadata['device_id'], (list, tuple)):
            op.metadata['transformer'] = op.metadata['device'] + op.metadata['device_id'][0]
            for id in op.metadata['device_id']:
                transformer = op.metadata['device'] + str(id)
                self.hetr.register_transformer(transformer)
        else:
            op.metadata['transformer'] = transformer
            self.hetr.register_transformer(transformer)

        if isinstance(op, TensorValueOp):
            op.states_read[0].metadata.update(op.metadata)


class CommunicationPass(GraphBuildingPass):
    def __init__(self, send_nodes):
        super(CommunicationPass, self).__init__()
        self.send_nodes = send_nodes

    def visit(self, op):
        args = list()
        for arg in op.args:
            node_type = get_node_type(from_node=arg, to_node=op)
            if node_type:
                pair = CommNodePair(from_node=arg, to_node=op, node_type=node_type)
                send_node, recv_node = pair.get_nodes()
                self.send_nodes.add(send_node)
                args.append(recv_node)
            else:
                args.append(arg)

        op._args = tuple(args)

        # invalidate deps cache as op._args is updated
        op.invalidate_property_cache('all_deps')

    def do_pass(self, ops, transformer):
        super(CommunicationPass, self).do_pass(ops, transformer)
        ops.update(self.send_nodes)


class DistributedPass(GraphBuildingPass):
    """
    DistributedPass clones subgraphs of which root is a GatherSendOp to finish
    implementing scatter/gather. It assigns new parallel axes and device_id

    Assumes
        CommunicationPass ran already, to insert incomplete Scatter/Gather nodes
        metadata['parallel', 'device_id', ] are present on nodes
    """

    def __init__(self, send_nodes):
        super(DistributedPass, self).__init__()
        self.send_nodes = send_nodes
        self.num_devices = 0

    def do_pass(self, ops, transformer):

        ops = OrderedSet(op.forwarded for op in ops)

        for op in reversed(Op.ordered_ops(ops)):
            if op.metadata.get('marker') == 'gather':
                # op is GatherRecvOp
                self.parallel_axes = op.metadata['parallel']

                gather_send_op = op.send_node()
                new_axes = calculate_new_axes(
                    gather_send_op.axes, self.parallel_axes, len(op.from_id))

                # clone nodes for each device_id
                for i, id in enumerate(op.from_id):
                    new_gather_send_op = clone_graph(root=gather_send_op, device_id=id,
                                                     shared_queues_idx=i, axes=new_axes)
                    self.send_nodes.add(new_gather_send_op)

                self.send_nodes.remove(gather_send_op)
                # how to make sure this part of the graph is not working?
                for o in Op.ordered_ops([gather_send_op]):
                    o.metadata['transformer'] = None
