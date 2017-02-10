from __future__ import division
from ngraph.op_graph.op_graph import make_axes, make_axis


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
        new_node.args = (new_arg,)
        new_node.dtype = node.dtype
        new_node.metadata['device'] = node.metadata['device']
        new_node.metadata['device_id'] = device_id
        node.metadata['device_id'] = node.metadata['device_id'][0]

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

    else:
        new_node = node.__class__()
        if node.initializers is not None:
            for initializer in node.initializers:
                new_initializer = initializer.__class__(
                    tensor=new_node, valfun=initializer.valfun)
                new_node.add_initializer(new_initializer)
        new_node._TensorOp__axes = new_axes
        new_node.args = node.args
        new_node.dtype = node.dtype
        new_node.metadata['device'] = node.metadata['device']
        new_node.metadata['device_id'] = device_id

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
