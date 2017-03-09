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

        def set_new_axes(node):
            if hasattr(node, 'axes'):
                node._TensorOp__axes = self.new_axes

        ops = OrderedSet(op.forwarded for op in ops)

        for op in reversed(Op.ordered_ops(ops)):
            if op.metadata.get('marker') == 'gather':
                self.parallel_axes = op.metadata['parallel']

                self.new_axes = calculate_new_axes(
                    op.send_node().axes, self.parallel_axes, len(op.from_id), False)

                nodes_to_clone = Op.ordered_ops([op.send_node()])
                map(set_new_axes, nodes_to_clone)

                # clone nodes for other device_id
                for i, id in enumerate(op.from_id[1:], start=1):
                    # compute the axes for last device
                    if i == (len(op.from_id) - 1):
                        self.new_axes = calculate_new_axes(
                            op.axes, self.parallel_axes, len(op.from_id), True)

                    # print('device_id={}, device_idx={}'.format(id, i))
                    # print('nodes to clone: {}\n'.format(nodes_to_clone))
                    cloned_nodes = self.clone_nodes(nodes=nodes_to_clone, device_id=id,
                                     device_idx=i, new_axes=self.new_axes)
                    # print('cloned nodes: {}\n'.format(cloned_nodes))


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
