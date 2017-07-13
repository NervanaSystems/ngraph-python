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
import socket

from orderedset import OrderedSet

from ngraph.factory.comm_node_factory import get_comm_pattern, CommNodePair
from ngraph.op_graph.op_graph import Op, TensorValueOp
from ngraph.transformers.hetr.hetr_utils import clone_graph
from ngraph.transformers.passes.passes import GraphBuildingPass


class DeviceAssignPass(GraphBuildingPass):

    def __init__(self, hetr, default_device, default_device_id, **kwargs):
        super(DeviceAssignPass, self).__init__(**kwargs)
        self.hetr = hetr
        self.default_device = default_device
        self.default_device_id = default_device_id

    def visit(self, op, *args):
        device = op.metadata.setdefault('device', self.default_device)
        if 'device_id' in op.metadata and \
           isinstance(op.metadata['device_id'], (list, tuple)) and \
           len(op.metadata['device_id']) == 1:
            op.metadata['device_id'] = op.metadata['device_id'][0]
        device_id = op.metadata.setdefault('device_id', self.default_device_id)
        transformer = "{}{}".format(device, device_id)
        op.metadata['host_transformer'] = socket.gethostname()
        if isinstance(op.metadata['device_id'], (list, tuple)):
            op.metadata['transformer'] = \
                [op.metadata['device'] + str(i) for i in op.metadata['device_id']]
            [self.hetr.register_transformer(tname) for tname in op.metadata['transformer']]
        else:
            op.metadata['transformer'] = transformer
            self.hetr.register_transformer(transformer)

        if isinstance(op, TensorValueOp):
            op.states_read[0].metadata.update(op.metadata)


class CommunicationPass(GraphBuildingPass):

    def __init__(self, send_nodes, **kwargs):
        super(CommunicationPass, self).__init__(**kwargs)
        self.send_nodes = send_nodes

    def visit(self, op, *op_args):
        args = list()
        for arg in op_args:
            comm_pattern = get_comm_pattern(from_node=arg, to_node=op)
            if comm_pattern:
                pair = CommNodePair(from_node=arg, to_node=op, node_type=comm_pattern)
                if pair.get_send_node():
                    self.send_nodes.add(pair.get_send_node())
                if pair.get_recv_node():
                    args.append(pair.get_recv_node())
            else:
                args.append(arg)

        op._args = tuple(args)

        # invalidate deps cache as op._args is updated
        op.invalidate_property_cache('all_deps')

    def do_pass(self, ops, **kwargs):
        super(CommunicationPass, self).do_pass(ops=ops, **kwargs)
        ops.update(self.send_nodes)


class DistributedPass(GraphBuildingPass):
    """
    DistributedPass clones subgraphs of which root is a GatherSendOp to finish
    implementing scatter/gather. It assigns new parallel axes and device_id

    Assumes
        CommunicationPass ran already, to insert incomplete Scatter/Gather nodes
        metadata['parallel', 'device_id', ] are present on nodes
    """

    def __init__(self, send_nodes, **kwargs):
        super(DistributedPass, self).__init__(**kwargs)
        self.send_nodes = send_nodes
        self.num_devices = 0

    def do_pass(self, ops, **kwargs):

        ops = OrderedSet(op.forwarded for op in ops)

        for op in reversed(Op.ordered_ops(ops)):
            if op.metadata.get('marker') == 'gather':
                # op is GatherRecvOp
                self.parallel_axes = op.metadata['parallel']
                gather_send_op = op.send_nodes[0]

                # clone nodes for each device_id
                replaced_send_ops = OrderedSet()
                new_gather_send_nodes = OrderedSet()
                for i, id in enumerate(op.from_id):
                    new_gather_send_op, new_sends, replaced_sends = clone_graph(
                        root=gather_send_op,
                        clone_id=id,
                        shared_queues_idx=i,
                        parallel_axis=self.parallel_axes,
                        num_clones=len(op.from_id))

                    new_gather_send_nodes.add(new_gather_send_op)

                    new_sends.add(new_gather_send_op)
                    for o in new_sends:
                        self.send_nodes.add(o)

                    replaced_send_ops |= replaced_sends

                op.send_nodes = new_gather_send_nodes

                replaced_send_ops.add(gather_send_op)
                for o in replaced_send_ops:
                    self.send_nodes.remove(o)
