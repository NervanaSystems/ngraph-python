from passes import PeepholeGraphPass
from ngraph.op_graph.communication import Send
from ngraph.op_graph.communication import Recv
from ngraph.op_graph.communication import Gather_Send, Gather_Recv, Scatter_Send, Scatter_Recv
from ngraph.util.ordered import OrderedSet
from ngraph.op_graph.op_graph import Op
from ngraph.util.hetr_utils import clone, calculate_new_axes
import multiprocessing
import collections


class DeviceAssignPass(PeepholeGraphPass):

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


class CommunicationPass(PeepholeGraphPass):

    def __init__(self, send_nodes, scatter_shared_queues, gather_shared_queues):
        super(CommunicationPass, self).__init__()
        self.send_nodes = send_nodes
        self.scatter_shared_queues = scatter_shared_queues
        self.gather_shared_queues = gather_shared_queues

    def insert_scatter_nodes(self, op, scatter_from, args):
        assert 'parallel' in op.metadata, "Fatal: 'parallel' hint not provided!"

        if scatter_from.__class__.__name__ is 'BroadcastOp':
            if scatter_from.args[0].is_constant:
                args.append(scatter_from)

        elif not scatter_from.is_constant:
            for i in range(len(op.metadata['device_id'])):
                self.scatter_shared_queues.append(multiprocessing.Queue())

            scatter_from.metadata['marker'] = 'scatter'

            scatter_send_node = Scatter_Send(from_node=scatter_from,
                                             axes=scatter_from.axes,
                                             parallel_axis=op.metadata['parallel'],
                                             queues=self.scatter_shared_queues,
                                             device=scatter_from.metadata['device'],
                                             device_id=scatter_from.metadata['device_id'],
                                             to_id=op.metadata['device_id'])

            scatter_recv_node = Scatter_Recv(axes=scatter_from.axes, dtype=scatter_from.dtype,
                                             queue=self.scatter_shared_queues[0],
                                             send_node=scatter_send_node,
                                             device=op.metadata['device'],
                                             device_id=op.metadata['device_id'][0])

            self.send_nodes.append(scatter_send_node)
            args.append(scatter_recv_node)

        else:
            args.append(scatter_from)

    def insert_gather_nodes(self, op, gather_from, args):
        assert 'parallel' in gather_from.metadata, "Fatal: 'parallel' hint not provided!"

        for i in range(len(gather_from.metadata['device_id'])):
            self.gather_shared_queues.append(multiprocessing.Queue())

        gather_send_node = Gather_Send(from_node=gather_from,
                                       axes=gather_from.axes,
                                       queue=self.gather_shared_queues[0],
                                       device=gather_from.metadata['device'],
                                       device_id=gather_from.metadata['device_id'][0])

        gather_recv_node = Gather_Recv(axes=gather_from.axes, dtype=gather_from.dtype,
                                       parallel_axis=gather_from.metadata['parallel'],
                                       queues=self.gather_shared_queues,
                                       send_node=gather_send_node,
                                       device=gather_from.metadata['device'],
                                       device_id=op.metadata['device_id'],
                                       from_id=gather_from.metadata['device_id'])

        self.send_nodes.append(gather_send_node)
        args.append(gather_recv_node)

    def insert_send_recv_nodes(self, op, arg, args):
        shared_queue = multiprocessing.Queue()
        self.send_nodes.append(Send(from_node=arg, queue=shared_queue,
                                    device=arg.metadata['device'],
                                    device_id=arg.metadata['device_id']))

        args.append(Recv(axes=arg.axes, dtype=arg.dtype, queue=shared_queue,
                         send_node=self.send_nodes[-1],
                         device=op.metadata['device'],
                         device_id=op.metadata['device_id']))

    def visit(self, op):
        args = list()
        for arg in op.args:
            if isinstance(op.metadata['device_id'], (list, tuple)):
                self.insert_scatter_nodes(op, arg, args)
            elif isinstance(arg.metadata['device_id'], (list, tuple)):
                self.insert_gather_nodes(op, arg, args)
            elif op.metadata['device_id'] != arg.metadata['device_id']:
                self.insert_send_recv_nodes(op, arg, args)
            elif op.metadata['device'] != arg.metadata['device']:
                self.insert_send_recv_nodes(op, arg, args)
            else:
                args.append(arg)

        if isinstance(op.args, tuple):
            op.args = tuple(args)
        else:
            op.args(args)  # setter is called args

    def do_pass(self, ops, inits):
        ops, inits = super(CommunicationPass, self).do_pass(ops, inits)
        ops.update(self.send_nodes)
        return ops, inits


class DistributedPass(PeepholeGraphPass):

    def __init__(self, send_nodes, scatter_shared_queues, gather_shared_queues):
        super(DistributedPass, self).__init__()
        self.send_nodes = send_nodes
        self.scatter_shared_queues = scatter_shared_queues
        self.gather_shared_queues = gather_shared_queues
        self.num_devices = 0

    def do_traversal(self, root):
        # Note: This is almost identical to Op's visit_input_closure.
        available = OrderedSet()
        counts = dict()
        parents = collections.defaultdict(list)
        ready = OrderedSet()
        nodes = list()

        available.add(root)
        while available:
            node = available.pop()
            node.update_forwards()

            if node in counts:
                continue

            children = [child.forwarded for child in node.all_deps]
            if children:
                counts[node] = len(children)
                for child in children:
                    parents[child].append(node)
                available.update(children)
            else:
                ready.add(node)

        while ready:
            node = ready.pop()
            nodes.append(node)
            for p in parents.get(node, []):
                count = counts[p] - 1
                if count == 0:
                    ready.add(p)
                    del counts[p]
                else:
                    counts[p] = count

        return nodes

    def clone_nodes(self, nodes, device_id, new_axes, scatter_shared_queue, gather_shared_queue):
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
                        new_node = clone(node=arg, new_axes=new_axes,
                                         scatter_shared_queue=scatter_shared_queue,
                                         gather_shared_queue=gather_shared_queue,
                                         device_id=device_id)
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
                                      scatter_shared_queue=scatter_shared_queue,
                                      gather_shared_queue=gather_shared_queue,
                                      send_nodes=self.send_nodes, arg1=subgraph[-1]))
                elem = elem + 1

        return subgraph

    def do_pass(self, ops, inits):
        ops = OrderedSet(op.forwarded for op in ops)

        def set_new_axes(root, num_devices):
            visit = self.do_traversal(root)
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

                            nodes = self.do_traversal(arg.send_node())
                            self.clone_nodes(nodes, arg.from_id[d], self.new_axes,
                                             self.scatter_shared_queues[d],
                                             self.gather_shared_queues[d])

                args.append(arg)

            if isinstance(op.args, tuple):
                op.args = tuple(args)
            else:
                op.args(args)
        return ops, inits


class ChildTransformerPass(PeepholeGraphPass):

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
