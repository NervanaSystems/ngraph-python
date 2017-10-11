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

from ngraph.op_graph.axes import Axes, make_axis
from ngraph.op_graph.comm_nodes import set_parallel_axes
from ngraph.op_graph.op_graph import Op, DotOp, TensorValueOp, SequentialOp, ParallelOp
from ngraph.op_graph.comm_nodes import GatherSendOp, RecvOp, ScatterSendOp, ScatterRecvOp, \
    GPUQueueRecvOp, CPUMlslSendOp, AllReduceOp, BroadcastRecvOp, GatherRecvOp
from orderedset import OrderedSet
from ngraph.op_graph.serde.serde import serialize_graph, deserialize_graph

import uuid
import collections


def get_iterable(x):
    if isinstance(x, collections.Iterable) and not isinstance(x, str):
        return x
    else:
        return (x,)


def comm_path_exists(fro, to):
    """
    Find a path from fro to to, including paths non-explicit edges from
    a Receiver to its Sender.

    Note- this is a non-standard traversal, as most traversals stop at a Receiver.
    """

    # TODO: Issue #1865 does this correctly handle traversing multiple send-recv junctions
    # from fro to to?

    visit = OrderedSet(fro.args)
    visit.add(fro)
    while visit:
        v = visit.pop()
        if v == to:
            return True
        if isinstance(v, RecvOp):
            visit |= get_iterable(v.send_node())
        elif isinstance(v, (SequentialOp, ParallelOp)):
            visit.update(v.control_deps)
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
            visit |= get_iterable(v.send_node())
        elif isinstance(v, (SequentialOp, ParallelOp)):
            visit.update(v.control_deps)
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
                if not r.metadata['transformer'] == op.metadata['transformer']:
                    continue
                send_nodes = get_iterable(r.send_node())
                for s in send_nodes:
                    if comm_path_exists(fro=s, to=op):
                        r.add_control_dep(op)


def clone_graph(root, clone_id, parallel_axis):
    """
    clone graph with serde (serialization)
    input:
    output: new_root of the cloned graph
    """
    
    # clone nodes with GatherSendOp as root using serde
    ser_cloned_nodes = deserialize_graph(serialize_graph([root]))

    new_root = next((o for o in ser_cloned_nodes if o.uuid == root.uuid), None)

    orig_ops = {op.uuid: op for op in Op.ordered_ops([root])}
    cloned_graph = Op.ordered_ops([new_root])

    new_send_nodes = OrderedSet()
    replaced_send_nodes = OrderedSet()

    # update newly cloned op metadata, generate new UUIDs
    for op in cloned_graph:
        cloned_ops = orig_ops[op.uuid].metadata.get('clones')
        if cloned_ops is None or cloned_ops.get(str(clone_id)) is None:
            op.metadata['transformer'] = op.metadata['device'] + str(clone_id)
            op.metadata['device_id'] = str(clone_id)

            if isinstance(op, (ScatterRecvOp, GatherSendOp, AllReduceOp, BroadcastRecvOp)):
                # for gpu communication op buffer
                op.idx = int(clone_id)
                if isinstance(op, (ScatterRecvOp, BroadcastRecvOp)):
                    op._send_node = orig_ops[op.uuid].send_node()

            if hasattr(op, 'reduction_axes') and parallel_axis in op.reduction_axes:
                op.reduction_axes = set_parallel_axes(op.reduction_axes, parallel_axis)

            if getattr(op, 'axes', None) is not None \
                    and parallel_axis in Axes.as_flattened_list(op.axes):
                # if parallel_axis in Axes.as_flattened_list(op.axes):
                op._axes = set_parallel_axes(op.axes, parallel_axis)
                if isinstance(op, DotOp):
                    if parallel_axis in op.x_out_axes:
                        op.x_out_axes = set_parallel_axes(op.x_out_axes,
                                                          parallel_axis)
                    elif parallel_axis in op.y_out_axes:
                        op.y_out_axes = set_parallel_axes(op.y_out_axes,
                                                          parallel_axis)
                    else:
                        raise ValueError("Missing parallel_axis in Op's "
                                         "x_out_axes or y_out_axes")

            if isinstance(op, TensorValueOp) and parallel_axis in op.tensor.axes:
                op.tensor._axes = set_parallel_axes(op.tensor.axes, parallel_axis)

            args_list = list(op.args)
            for arg_idx, arg_op in enumerate(args_list):
                if arg_op.uuid in orig_ops.keys():
                    if orig_ops[arg_op.uuid].metadata.get('clones') and \
                       orig_ops[arg_op.uuid].metadata['clones'].get(str(clone_id)):
                        args_list[arg_idx] = \
                            orig_ops[arg_op.uuid].metadata['clones'].get(str(clone_id))

            op.invalidate_property_cache('all_deps')
            op._args = tuple(args_list)
            if op != new_root:
                if orig_ops[op.uuid].metadata.get('clones') is None:
                    orig_ops[op.uuid].metadata['clones'] = dict()
                    orig_ops[op.uuid].metadata['clones'][str(clone_id)] = op
                else:
                    orig_ops[op.uuid].metadata['clones'][str(clone_id)] = op

            op.uuid = uuid.uuid4()

    # create new uuids for all the ops that have references to the new root
    for _op in Op.all_op_references([new_root]):
        _op.uuid = uuid.uuid4()

    return new_root, new_send_nodes, replaced_send_nodes


def update_parallel_axis(root, parallel_axis):
    for op in Op.ordered_ops([root]):

        if hasattr(op, 'reduction_axes') and parallel_axis in op.reduction_axes:
            op.reduction_axes = set_parallel_axes(op.reduction_axes, parallel_axis)

        if getattr(op, 'axes', None) is not None \
                and parallel_axis in Axes.as_flattened_list(op.axes):
            # if parallel_axis in Axes.as_flattened_list(op.axes):
            op._axes = set_parallel_axes(op.axes, parallel_axis)
            if isinstance(op, DotOp):
                if parallel_axis in op.x_out_axes:
                    op.x_out_axes = set_parallel_axes(op.x_out_axes,
                                                      parallel_axis)
                elif parallel_axis in op.y_out_axes:
                    op.y_out_axes = set_parallel_axes(op.y_out_axes,
                                                      parallel_axis)
                else:
                    raise ValueError("Missing parallel_axis in Op's "
                                     "x_out_axes or y_out_axes")

        if isinstance(op, TensorValueOp) and parallel_axis in op.tensor.axes:
            op.tensor._axes = set_parallel_axes(op.tensor.axes, parallel_axis)


def update_ops_metadata(ops, device_idx):
    """
    Description:
        Since we no longer clone the graph,
        We need to update the transformer and device_id metadata for the ops before they are distributed to each device
        The transformer metadata is used to identify which computation ops are needed for each device
        The device_id metadata is used as a parameter to identify each device when making mpi collective calls
    Input: list of ops
    Output:
    """
    
    ops = OrderedSet(op.forwarded for op in ops)
    for op in reversed(Op.ordered_ops(ops)):
        if op.metadata.get('marker') == 'gather':
            gather_send_op = op.send_nodes[0]

            for _op in Op.ordered_ops([gather_send_op]):
                _op.metadata['transformer'] = _op.metadata['device'] + str(op.from_id[device_idx])
                _op.metadata['device_id'] = str(op.from_id[device_idx])
                _op.idx = device_idx    # used by gpu in collective op kernels (tensor_ops.py)


def get_available_ports():
    import os
    if "HETR_SERVER_PORTS" in os.environ:
        return os.getenv("HETR_SERVER_PORTS")
    else:
        return ['52051', '52052', '52053', '52054', '52055', '52056', '52057', '52058']
