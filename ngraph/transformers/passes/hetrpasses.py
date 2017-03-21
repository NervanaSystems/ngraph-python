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
from ngraph.util.hetr_utils import clone, find_recvs
from ngraph.util.ordered import OrderedSet
from ngraph.op_graph.comm_nodes import GatherSendOp, ScatterRecvOp

from ngraph.op_graph.serde.serde import serialize_graph, deserialize_graph
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

    def clone_nodes(self, nodes, device_id, device_idx, new_axes):
        # TODO (wenzhe)implement with serde (serialization)
        subgraph = list()
        elem = 0

        # First find Add and then clone its args. This is needed to
        # make sure Add has the correct arguments at init/clone time.
        visit = nodes
        add_op_list = list()
        for v in visit:
            if v.__class__.__name__ is 'Add':
                add_op_list.append(visit.index(v))

        while visit:
            if len(add_op_list):
                for i in add_op_list:
                    v = visit[i]
                    for arg in v.args:
                        new_node = clone(node=arg, new_axes=new_axes,
                                         device_id=device_id,
                                         device_idx=device_idx)
                        subgraph.append(new_node)
                        visit.remove(arg)
                        elem = elem + 1
                    new_node = clone(node=v, new_axes=new_axes,
                                     device_id=device_id,
                                     arg1=subgraph[elem - 1],
                                     arg2=subgraph[elem - 2])
                    subgraph.append(new_node)
                    visit.remove(v)
                    add_op_list.pop(0)
            else:
                node = visit.pop()
                subgraph.append(
                    clone(node=node, new_axes=new_axes, device_id=device_id,
                          device_idx=device_idx, send_nodes=self.send_nodes,
                          arg1=subgraph[-1]))
                elem = elem + 1

        return subgraph

    def serde_clone_nodes(self, gather_send_op, device_id, device_idx, new_axes):
        """hacks to clone nodes with serialization"""

        # clone nodes with GatherSendOp as root using serialize_graph and deserialized_graph
        ser_str = serialize_graph([gather_send_op])
        ser_cloned_nodes = deserialize_graph(ser_str)

        # get the ops linked from ScatterRecvOp via args or control_deps
        new_gather_send_op = [o for o in ser_cloned_nodes if isinstance(o, GatherSendOp)]
        assert len(new_gather_send_op) == 1, 'multiple GatherSendOp after clone'

        new_gather_send_op = new_gather_send_op[0]
        # change self.send_nodes to reflect the clone
        self.send_nodes.add(new_gather_send_op)
        cloned_nodes = Op.ordered_ops([new_gather_send_op])

        # modify idx, axes, and other metadata eg. device_id, transformer
        for v in Op.ordered_ops([gather_send_op]):
            if isinstance(v, ScatterRecvOp):
                ScatterRecvOp_shared_queues = v.shared_queues
                send_node = v.send_node()
            elif isinstance(v, GatherSendOp):
                GatherSendOp_shared_queues = v.shared_queues

        assert ScatterRecvOp_shared_queues, 'ScatterRecvOp shared Qs is None'
        assert GatherSendOp_shared_queues, 'GatherSendOp shared Qs is None'

        for op in cloned_nodes:
            op.metadata['transformer'] = op.metadata['device'] + str(device_id)
            op.metadata['device_id'] = str(device_id)
            if op.metadata.get('host_transformer') == gather_send_op.metadata['host_transformer']:
                op.metadata['host_transformer'] = gather_send_op.metadata['host_transformer']
            if isinstance(op, ScatterRecvOp):
                op.shared_queues = ScatterRecvOp_shared_queues
                op._send_node = send_node
                op.idx = device_idx
            elif isinstance(op, GatherSendOp):
                op.shared_queues = GatherSendOp_shared_queues
                op.idx = device_idx
        return cloned_nodes

    def do_pass(self, ops, transformer):

        ops = OrderedSet(op.forwarded for op in ops)

        for op in reversed(Op.ordered_ops(ops)):
            if op.metadata.get('marker') == 'gather':
                # op is GatherGatherOp
                self.parallel_axes = op.metadata['parallel']

                new_axes = calculate_new_axes(
                    op.send_node().axes, self.parallel_axes,
                    len(op.from_id), False)
                Op.visit_input_closure(
                    [op.send_node()],
                    lambda x: setattr(x, '_axes', new_axes))

                nodes_to_clone = Op.ordered_ops([op.send_node()])
                # clone nodes for other device_id
                # todo: clone nodes for each device_id
                for i, id in enumerate(op.from_id[1:], start=1):
                    # get axes for last device if it's different
                    if i == (len(op.from_id) - 1) \
                            and self.parallel_axes.length % len(op.from_id) > 0:
                        new_axes = calculate_new_axes(
                            op.axes, self.parallel_axes, len(op.from_id), True)

                    # cloned_graph = self.clone_nodes(nodes=nodes_to_clone, device_id=id, device_idx=i, new_axes=new_axes)
                    ser_cloned_graph = self.serde_clone_nodes(gather_send_op=op.send_node(),
                                                              device_id=id,
                                                              device_idx=i,
                                                              new_axes=new_axes)
