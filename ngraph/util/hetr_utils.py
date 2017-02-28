from __future__ import division
from ngraph.op_graph.op_graph import make_axes, make_axis
from ngraph.op_graph.communication import Receiver
from ngraph.util.ordered import OrderedSet


def clone(
        node,
        new_axes,
        device_id,
        scatter_shared_queue=None,
        gather_shared_queue=None,
        send_nodes=None,
        arg1=None,
        arg2=None):
    if node.__class__.__name__ is 'AddOp':
        new_node = node.__class__(arg1, arg2)
        new_node._TensorOp__axes = new_axes
        new_node.dtype = node.dtype
        new_node.metadata['device'] = node.metadata['device']
        new_node.metadata['device_id'] = device_id
        node.metadata['device_id'] = node.metadata['device_id'][0]

    elif node.__class__.__name__ is 'BroadcastOp':
        new_arg = clone(node.args[0], new_axes, device_id)
        new_node = node.__class__(new_arg, new_axes)
        # new_node.args = (new_arg,)
        new_node.dtype = node.dtype
        new_node.metadata['device'] = node.metadata['device']
        new_node.metadata['device_id'] = device_id
        node.metadata['device_id'] = node.metadata['device_id'][0]

    elif node.__class__.__name__ is 'TensorValueOp':
        new_node = node.__class__(node.states_read[0])
        new_node.metadata['device'] = node.metadata['device']
        new_node.metadata['device_id'] = device_id


    elif node.__class__.__name__ is 'Scatter_Recv':
        new_node = node.__class__(
            axes=new_axes,
            dtype=node.dtype,
            queue=scatter_shared_queue,
            send_node=node.send_node(),
            device=node.metadata['device'],
            device_id=device_id)

    elif node.__class__.__name__ is 'Scatter_Send':
        pass

    elif node.__class__.__name__ is 'Gather_Send':
        new_node = node.__class__(
            from_node=arg1,
            axes=new_axes,
            queue=gather_shared_queue,
            device=node.metadata['device'],
            device_id=device_id)
        send_nodes.add(new_node)

    elif node.__class__.__name__ is 'Gather_Recv':
        pass

    elif 'marker' in node.metadata and node.metadata['marker'] is 'scatter':
        pass  # This node is marked to be scattered, so there is no need to clone it.

    elif  node.__class__.__name__ is 'AssignableTensorOp' and node.is_constant:
        new_node = node.__class__()
        if node.initializers is not None:
            for initializer in node.initializers:
                new_initializer = initializer.__class__(
                    tensor=new_node, valfun=initializer.valfun)
                new_node.add_initializer(new_initializer)
        new_node._TensorOp__axes = new_axes
        # new_node._TensorOp__args = node.args
        new_node.dtype = node.dtype
        new_node.metadata['device'] = node.metadata['device']
        new_node.metadata['device_id'] = device_id
    else:
        raise RuntimeError("Unsupported op type {} for clone.".format(node.__class__.__name__))

    return new_node


def calculate_new_axes(axes, parallel_axis, num_devices, is_last):
    new_axes = list()
    for a in axes:
        if parallel_axis == a:
            remainder = a.length % num_devices
            new_length = a.length // num_devices
            if remainder > 0:
                if is_last:
                    new_length += remainder
            new_axis = make_axis(new_length, a.name)
            new_axes.append(new_axis)
        else:
            new_axes.append(a)
    new_axes = make_axes(new_axes)
    return new_axes


def comm_path_exists(fro, to):
    """
    Find a path from fro to to, including paths non-explicit edges from
    a Receiver to its Sender.

    Note- this is a non-standard traversal, as most traversals stop at a Receiver.
    """

    # TODO: does this correctly handle traversing multiple send-recv junctions
    # from fro to to?

    visit = OrderedSet(fro.args)
    while visit:
        v = visit.pop()
        if v == to:
            return True
        if isinstance(v, Receiver):
            visit.add(v.send_node())
        else:
            visit.update(v.args)

    return False


def find_recvs(fro):
    # Find all the Receivers fro depends on
    visit = OrderedSet()
    recvs = OrderedSet()
    visit.add(fro)
    while visit:
        v = visit.pop()
        if isinstance(v, Receiver):
            recvs.add(v)
            visit.add(v.send_node())
        else:
            if hasattr(v, 'args'):
                visit.update(v.args)

    return recvs


def sort_ops_by_comm_deps(ops):
    """
    Sort the subgraphs identified by ops using communication dependencies.

    Find any Receiver nodes that an op depends on; add 'control_deps' from Receivers
    to any other op in ops which the Sender for that Receiver depends on.

    Ex.
    Whole Graph:
        X -> Send0
        Recv0 -> Y -> Send1
        Recv1 -> Z

    ops to be sorted:
        Send0, Z

    Deadlock would occur if Z ran before Send0, but there are no explicit edges
    connecting them.
    Using control_deps, the subgraph for this transformer looks like:

    X -> Send0 ====other_dep====> Recv1 -> Z

    This ensures that the built in logic in any child transformer, which sorts
    nodes based on control_deps,
    will produce a correct order if one is possible.
    """
    if len(ops) <= 1:
        return

    # For each return (ops), find out if there should be an other_dep added from any
    # other return to it based on communication dependencies
    ops_to_update = OrderedSet(ops)
    for op in ops_to_update:
        other_ops = set(ops) - set([op])
        for trav_op in other_ops:
            recvs = find_recvs(fro=trav_op)
            for r in recvs:
                if comm_path_exists(fro=r.send_node(), to=op):
                    if r.metadata['transformer'] == op.metadata['transformer']:
                        r.add_control_dep(op)
