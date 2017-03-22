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
from orderedset import OrderedSet
from ngraph.op_graph.comm_nodes import GatherSendOp, ScatterRecvOp
from ngraph.op_graph.serde.serde import serialize_graph, deserialize_graph
import uuid

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

    def clone_distributed_subgraph(self, root, device_id, device_idx, axes):
        """
        clone nodes with serde (serialization) module
        """

        # clone nodes with GatherSendOp as root using serde
        ser_cloned_nodes = deserialize_graph(serialize_graph([root]))
        cloned_root = next((o for o in ser_cloned_nodes if o.uuid == root.uuid), None)

        orig_ops = {op.uuid: op for op in Op.ordered_ops([root])}

        # Prune ops that are not control_deps of new_gather_send_op
        # deserialize includes extra referenced nodes
        cloned_graph = Op.ordered_ops([cloned_root])
        # update newly cloned op metadata, generate new UUIDs
        for op in cloned_graph:
            op.metadata['transformer'] = op.metadata['device'] + str(device_id)
            op.metadata['device_id'] = str(device_id)
            if isinstance(op, (ScatterRecvOp, GatherSendOp)):
                op.shared_queues = orig_ops[op.uuid].shared_queues
                op.idx = device_idx
                if isinstance(op, ScatterRecvOp):
                    op._send_node = orig_ops[op.uuid].send_node()

            # todo add distributed hetr tests where axes of last device is different
            if op._axes != axes:
                op._axes = axes

            op.uuid = uuid.uuid4()

        return cloned_root

    def do_pass(self, ops, transformer):

        ops = OrderedSet(op.forwarded for op in ops)

        for op in reversed(Op.ordered_ops(ops)):
            if op.metadata.get('marker') == 'gather':
                # op is GatherRecvOp
                self.parallel_axes = op.metadata['parallel']

                new_axes = calculate_new_axes(
                    op.send_node().axes, self.parallel_axes,
                    len(op.from_id), False)
                Op.visit_input_closure(
                    [op.send_node()],
                    lambda x: setattr(x, '_axes', new_axes))

                # clone nodes for other device_id
                # todo: clone nodes for each device_id
                for i, id in enumerate(op.from_id[1:], start=1):
                    # get axes for last device if it's different
                    if i == (len(op.from_id) - 1) \
                            and self.parallel_axes.length % len(op.from_id) > 0:
                        new_axes = calculate_new_axes(
                            op.axes, self.parallel_axes, len(op.from_id), True)

                    gather_send_op = op.send_node()
                    new_gather_send_op = self.clone_distributed_subgraph(
                        root=gather_send_op,
                        device_id=id,
                        device_idx=i,
                        axes=new_axes)
                    self.send_nodes.add(new_gather_send_op)
