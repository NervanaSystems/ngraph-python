from ngraph.transformers.passes.passes import GraphBuildingPass
from ngraph.factory.comm_nodes import calculate_new_axes
from ngraph.factory.comm_node_factory import get_node_type, CommNodePair
from ngraph.op_graph.op_graph import Op
from ngraph.util.hetr_utils import clone
from ngraph.util.ordered import OrderedSet
import collections
import socket


class DeviceAssignPass(GraphBuildingPass):

    def __init__(self, default_device, default_device_id, transformers):
        super(DeviceAssignPass, self).__init__()

        self.default_device = default_device
        self.default_device_id = default_device_id
        self.transformers = transformers

    def visit(self, op):
        device = op.metadata.setdefault('device', self.default_device)
        device_id = op.metadata.setdefault('device_id', self.default_device_id)
        transformer = "{}{}".format(device, device_id)
        op.metadata['transformer'] = transformer
        self.transformers.add(transformer)
        host_transformer = (socket.gethostname(), device_id)
        op.metadata['host_transformer'] = host_transformer


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

        op._Op__args = tuple(args)

    def do_pass(self, ops, transformer):
        super(CommunicationPass, self).do_pass(ops, transformer)
        ops.update(self.send_nodes)


class DistributedPass(GraphBuildingPass):

    def __init__(self, send_nodes):
        super(DistributedPass, self).__init__()
        self.send_nodes = send_nodes
        self.num_devices = 0

    def clone_nodes(self, nodes, device_id, device_idx, new_axes):
        subgraph = list()
        elem = 0

        # First find AddOp and then clone its args. This is needed to
        # make sure AddOp has the correct arguments at init/clone time.
        # TODO this might be needed for other ops as well.
        visit = nodes
        add_op_list = list()
        for v in visit:
            if v.__class__.__name__ is 'AddOp':
                add_op_list.append(visit.index(v))

        while visit:
            if len(add_op_list):
                for i in add_op_list:
                    v = visit[i]
                    for arg in v.args:
                        new_node = clone(node=arg, new_axes=new_axes, device_id=device_id,
                                         device_idx=device_idx)
                        subgraph.append(new_node)
                        visit.remove(arg)
                        elem = elem + 1
                    new_node = clone(node=v, new_axes=new_axes, device_id=device_id,
                                     arg1=subgraph[elem - 1], arg2=subgraph[elem - 2])
                    subgraph.append(new_node)
                    visit.remove(v)
                    add_op_list.pop(0)
            else:
                node = visit.pop()
                subgraph.append(clone(node=node, new_axes=new_axes, device_id=device_id,
                                      device_idx=device_idx, send_nodes=self.send_nodes,
                                      arg1=subgraph[-1]))
                elem = elem + 1

        return subgraph

    def do_pass(self, ops, transformer):
        ops = OrderedSet(op.forwarded for op in ops)

        def set_new_axes(root, num_devices):
            visit = Op.ordered_ops([root])
            self.new_axes = calculate_new_axes(root.axes, self.parallel_axis,
                                               num_devices, False)

            while visit:
                node = visit.pop()
                if hasattr(node, 'axes'):
                    node._TensorOp__axes = self.new_axes

        # Start traversal from the top to the bottom
        for op in reversed(Op.ordered_ops(ops)):
            args = list()
            for arg in op.args:
                if 'marker' in arg.metadata:
                    if 'gather' is arg.metadata['marker']:
                        self.parallel_axis = arg.metadata['parallel']
                        set_new_axes(arg.send_node(), len(arg.from_id))

                        for d in range(1, len(arg.from_id)):
                            if d == (len(arg.from_id) - 1):
                                self.new_axes = calculate_new_axes(arg.axes, self.parallel_axis,
                                                                   len(arg.from_id), True)

                            nodes = Op.ordered_ops([arg.send_node()])

                            self.clone_nodes(nodes=nodes, device_id=arg.from_id[d],
                                             device_idx=d, new_axes=self.new_axes)

                args.append(arg)

            if isinstance(op.args, tuple):
                op._Op__args = tuple(args)
            else:
                op.args(args)


class ChildTransformerPass(GraphBuildingPass):

    def __init__(self, transformer_list):
        super(ChildTransformerPass, self).__init__()

        self.transformer_list = transformer_list

    def visit(self, op):
        if 'device_id' not in op.metadata:
            return
        if isinstance(op.metadata['device_id'], (list, tuple)):
            op.metadata['transformer'] = op.metadata['device'] + op.metadata['device_id'][0]
            for i in range(len(op.metadata['device_id'])):
                transformer = op.metadata['device'] + op.metadata['device_id'][i]
                if transformer not in self.transformer_list:
                    self.transformer_list.append(transformer)
        else:
            transformer = op.metadata['device'] + str(op.metadata['device_id'])
            op.metadata['transformer'] = transformer
            if transformer not in self.transformer_list:
                self.transformer_list.append(transformer)
