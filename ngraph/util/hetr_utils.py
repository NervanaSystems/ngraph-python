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
from ngraph.op_graph.op_graph import Op
from ngraph.op_graph.comm_nodes import SendOp, ScatterSendOp, GatherSendOp, RecvOp, \
    ScatterRecvOp, GatherRecvOp
from ngraph.util.ordered import OrderedSet
from ngraph.op_graph.serde.serde import serialize_graph, deserialize_graph
import uuid
import ngraph as ng


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


def clone_graph(root, device_id, shared_queues_idx, axes):
    """
    clone graph with serde (serialization)
    input:
    output: new_root of the cloned graph
    """

    # clone nodes with GatherSendOp as root using serde
    ser_cloned_nodes = deserialize_graph(serialize_graph([root]))
    new_root = next((o for o in ser_cloned_nodes if o.uuid == root.uuid), None)

    orig_ops = {op.uuid: op for op in Op.ordered_ops([root])}

    # Prune ops that are not control_deps of new_gather_send_op
    # deserialize includes extra referenced nodes
    cloned_graph = Op.ordered_ops([new_root])
    # update newly cloned op metadata, generate new UUIDs
    for op in cloned_graph:
        op.metadata['transformer'] = op.metadata['device'] + str(device_id)
        op.metadata['device_id'] = str(device_id)
        if isinstance(op, (ScatterRecvOp, GatherSendOp)):
            op._shared_queues = orig_ops[op.uuid].shared_queues
            op.idx = shared_queues_idx
            if isinstance(op, ScatterRecvOp):
                op._send_node = orig_ops[op.uuid].send_node()

        op._axes = axes
        op.uuid = uuid.uuid4()

    return new_root


def create_send_recv_graph():
    axes = ng.make_axes([ng.make_axis(length=10, name='A'), ng.make_axis(length=15, name='B')])

    with ng.metadata(device=None, device_id=None, transformer=None, host_transformer=None):
        from_node = ng.placeholder(axes)
        to_node = ng.placeholder(axes)
    send_x = SendOp(from_node=from_node)
    recv_x = RecvOp(to_node=to_node, send_node=send_x)

    with ng.metadata(device=None, device_id=None, transformer=None, host_transformer=None):
        x_plus_one = recv_x + 1

    send_x_plus_one = SendOp(from_node=x_plus_one)
    recv_x_plus_one = RecvOp(to_node=to_node, send_node=send_x_plus_one)

    with ng.metadata(device=None, device_id=None, transformer=None, host_transformer=None):
        z = recv_x_plus_one + 2
    return z, recv_x, recv_x_plus_one, send_x, x_plus_one, from_node, send_x_plus_one


def create_scatter_gather_graph():
    ax_a = ng.make_axis(length=10, name='A')
    ax_b = ng.make_axis(length=20, name='B')
    axes = ng.make_axes([ax_a, ax_b])

    with ng.metadata(parallel=ax_b, device=(0, 1), device_id=(0, 1),
                     transformer=None, host_transformer=None):
        from_node = ng.placeholder(axes)
        to_node = ng.placeholder(axes)
    scatter_send_x = ScatterSendOp(from_node=from_node, to_node=to_node)
    scatter_recv_a = ScatterRecvOp(to_node=to_node, send_node=scatter_send_x)
    scatter_recv_b = ScatterRecvOp(to_node=to_node, send_node=scatter_send_x)
    gather_send_a = GatherSendOp(from_node=scatter_recv_a)
    gather_send_b = GatherSendOp(from_node=scatter_recv_b)
    gather_recv_x_plus_one = GatherRecvOp(from_node=from_node, to_node=to_node,
                                          send_node=gather_send_a)
    return scatter_send_x, scatter_recv_a, scatter_recv_b, \
        gather_send_a, gather_send_b, gather_recv_x_plus_one
