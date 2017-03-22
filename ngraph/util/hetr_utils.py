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
from __future__ import division
from ngraph.op_graph.op_graph import AssignableTensorOp, BroadcastOp, \
    TensorValueOp
from ngraph.factory.comm_nodes import GatherSendOp, GatherRecvOp, \
    RecvOp, ScatterSendOp, ScatterRecvOp
from orderedset import OrderedSet


def clone(
        node,
        new_axes,
        device_id,
        device_idx=None,
        send_nodes=None,
        arg1=None,
        arg2=None):
    if node.__class__.__name__ is 'Add':
        new_node = node.__class__(arg1, arg2)
        new_node._axes = new_axes
        new_node.dtype = node.dtype
        new_node.metadata['device'] = node.metadata['device']
        new_node.metadata['device_id'] = device_id
        new_node.metadata['transformer'] = node.metadata['device'] + str(device_id)
        node.metadata['device_id'] = node.metadata['device_id'][0]
        new_node.metadata['host_transformer'] = node.metadata['host_transformer']

    elif isinstance(node, BroadcastOp):
        new_arg = clone(node.args[0], new_axes, device_id)
        new_node = node.__class__(new_arg, new_axes)
        new_node.dtype = node.dtype
        new_node.metadata['device'] = node.metadata['device']
        new_node.metadata['device_id'] = device_id
        new_node.metadata['transformer'] = node.metadata['device'] + str(device_id)

        node.metadata['device_id'] = node.metadata['device_id'][0]
        new_node.metadata['host_transformer'] = node.metadata['host_transformer']

    elif isinstance(node, TensorValueOp):
        new_node = node.__class__(node.states_read[0])
        new_node.metadata['device'] = node.metadata['device']
        new_node.metadata['device_id'] = device_id
        new_node.metadata['host_transformer'] = node.metadata['host_transformer']
        new_node.metadata['transformer'] = node.metadata['device'] + str(device_id)

    elif isinstance(node, ScatterRecvOp):
        new_node = node.__class__(
            to_node=node,
            send_node=node.send_node(),
            device_idx=device_idx)
        new_node.metadata['transformer'] = node.metadata['device'] + str(device_id)

    elif isinstance(node, ScatterSendOp):
        pass

    elif isinstance(node, GatherSendOp):
        new_node = node.__class__(
            from_node=arg1,
            clone_node=node,
            device_idx=device_idx)
        new_node.metadata['transformer'] = node.metadata['device'] + str(device_id)
        send_nodes.add(new_node)

    elif isinstance(node, GatherRecvOp):
        pass

    elif 'marker' in node.metadata and node.metadata['marker'] is 'scatter':
        pass  # This node is marked to be scattered, so there is no need to clone it.

    elif isinstance(node, AssignableTensorOp) and node.is_constant:
        new_node = node.__class__()
        new_node.initial_value = node.initial_value
        new_node._axes = new_axes
        new_node.dtype = node.dtype
        new_node.metadata['device'] = node.metadata['device']
        new_node.metadata['device_id'] = device_id
        new_node.metadata['transformer'] = node.metadata['device'] + str(device_id)

    else:
        raise RuntimeError("Unsupported op type {} for clone.".format(node.__class__.__name__))

    return new_node


def comm_path_exists(fro, to):
    """
    Find a path from fro to to, including paths non-explicit edges from
    a Receiver to its Sender.

    Note- this is a non-standard traversal, as most traversals stop at a Receiver.
    """

    # TODO: does this correctly handle traversing multiple send-recv junctions
    # from fro to to?

    visit = OrderedSet(fro.args)
    visit.add(fro)
    while visit:
        v = visit.pop()
        if v == to:
            return True
        if isinstance(v, RecvOp):
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
        if isinstance(v, RecvOp):
            recvs.add(v)
            visit.add(v.send_node())
        else:
            if hasattr(v, 'args'):
                visit.update(v.args)

    return recvs


def update_comm_deps(ops):
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
