# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
"""
This module handles ngraph IR topology serialization. The general approach for serialization is to

1. Take in a list of ngraph Ops, and use the new Op.all_op_references method to grab (recursively)
all references in the Ops attributes (__dict__ keys), for ops.
2. For each op in this list, convert each to a protobuf object by again iterating through
attributes and converting each to a protobuf object and adding it to the OpAttr protobuf map. We
ignore any attributes of an op that reference another Op or Ops (these are handled later by the
`add_edges` function).
3. For each op obtained in 2, iterate through all attributes for the last time and for any that
references an op or list of ops, create an protobuf edge using the UUIDs of the head and tail ops.
Depending on the attribute name, encode the edge with a special type, or edge attribute to indicate
where it should be deserialized into.
4. Return the list of ops and edges in a GraphDef protobuf serialized string.

Deserialization happens in roughly the same order: first create Python ops, then hook up references
between them.

Currently only python public (aka non underscore prefixed) attributes are referenced with the
exception of those in EXCEPTION_ATTRIBUTES and starting with `_is_`.
"""
import uuid
import weakref
import pkgutil
import importlib
from collections import Iterable
from builtins import map

import six
import numpy as np

import ngraph
import ngraph.op_graph.op_graph as op_graph
from ngraph.op_graph.op_graph import Op
from ngraph.op_graph.axes import Axes, Axis, FlattenedAxis
from ngraph.util.ordered import OrderedSet

import ngraph.op_graph.serde.ops_pb2 as ops_pb


# Attributes of an Op that are private but we want to serialize
EXCEPTION_ATTRIBUTES = {'_axes', '_tensor', '_const', '_deriv_handler'}

# Dict of Axis and Axes UUID to Axis to enable matching of deserialized axis
GLOBAL_AXIS_REGISTRY = weakref.WeakValueDictionary()


##################
# SERIALIZATION
##################


def dtype_to_protobuf(numpy_dtype):
    return getattr(ops_pb, numpy_dtype.name.upper())


def axis_to_protobuf(axis):
    pb_axis = ops_pb.Axis()
    pb_axis.name = axis.name

    if isinstance(axis, FlattenedAxis):
        pb_axis.flattened_axes.CopyFrom(axes_to_protobuf(axis._axes))
        pb_axis.length = axis.length
        pb_axis.uuid.uuid = axis.uuid.bytes
        return pb_axis

    pb_axis.length = axis.length
    pb_axis.batch = axis.is_batch
    pb_axis.recurrent = axis.is_recurrent
    pb_axis.uuid.uuid = axis.uuid.bytes

    return pb_axis


def axes_to_protobuf(axes):
    pb_axes = ops_pb.Axes()
    for axis in axes:
        pb_axis = pb_axes.axes.add()
        pb_axis.CopyFrom(axis_to_protobuf(axis))
    pb_axes.uuid.uuid = axes.uuid.bytes
    return pb_axes


def tensor_to_protobuf(tensor):
    pb_tensor = ops_pb.Tensor()
    pb_tensor.dtype = dtype_to_protobuf(tensor.dtype)
    pb_tensor.shape.extend(tensor.shape)
    if tensor.dtype == np.float16:
        destination = pb_tensor.float_data
    elif tensor.dtype == np.float32:
        destination = pb_tensor.float_data
    elif tensor.dtype == np.float64:
        destination = pb_tensor.double_data
    elif tensor.dtype == np.int64:
        destination = pb_tensor.int_data
    elif tensor.dtype == np.uint64:
        destination = pb_tensor.uint_data

    if isinstance(tensor, np.ndarray):
        destination.extend(tensor.ravel().tolist())
    elif isinstance(tensor, np.generic):
        # A numpy scalar (obtained with `my_ndarray[()]`)
        destination.append(np.asscalar(tensor))
    else:
        raise ValueError("Unknown tensor value of {}".format(tensor))
    return pb_tensor


def unhandled_scalar_value(value):
    return ValueError("Unable to convert item type {} to protobuf for item {}".format(type(value),
                      value))


def is_scalar_type(value):
    return value is None or \
        isinstance(value, (str, six.text_type, float, bool, dict, slice, np.generic)
                   + six.integer_types)


def assign_scalar(message, value):
    """
    Adds the appropriate scalar type of value to the protobuf message
    """
    if value is None:
        message.null_val = True
    elif isinstance(value, np.generic):
        assign_scalar(message, np.asscalar(value))
    elif isinstance(value, (str, six.text_type)):
        message.string_val = value
    elif isinstance(value, np.dtype):
        message.dtype_val = dtype_to_protobuf(value)
    elif isinstance(value, float):
        message.double_val = value
    elif isinstance(value, bool):
        message.bool_val = value
    elif isinstance(value, six.integer_types):
        message.int_val = value
    elif isinstance(value, slice):
        slice_val = ops_pb.Slice()
        if value.start is not None:
            slice_val.start.value = value.start
        if value.step is not None:
            slice_val.step.value = value.step
        if value.stop is not None:
            slice_val.stop.value = value.stop
        message.slice_val.CopyFrom(slice_val)
    elif isinstance(value, dict):
        for key in value:
            assign_scalar(message.map_val.map[key], value[key])
        # This encodes an empty dict for deserialization
        assign_scalar(message.map_val.map['_ngraph_map_sentinel_'], '')
    else:
        raise unhandled_scalar_value(value)


def assign_op_attr(message, value):
    """
    Assigns a python object in value to the protobuf object `message` after conversion to
    the equivalent protobuf object.

    Args:
        message <protobuf OpAttr>: protobuf object to have value assigned to after conversion
            to protobuf.
        value <python object>: The python object to be converted and assigned.
    """
    if is_scalar_type(value):
        assign_scalar(message.scalar, value)
    elif isinstance(value, Axes):
        message.axes.CopyFrom(axes_to_protobuf(value))
    elif isinstance(value, Axis):
        message.axis.CopyFrom(axis_to_protobuf(value))
    elif isinstance(value, np.ndarray):
        message.tensor.CopyFrom(tensor_to_protobuf(value))
    elif isinstance(value, Iterable):
        if len(value) > 0:
            for item in value:
                if is_scalar_type(item):
                    assign_scalar(message.repeated_scalar.val.add(), item)
                else:
                    # raise unhandled_scalar_value(item)
                    print('skipped unhandled_scalar_value {} in serde.'.format(item))
        else:
            assign_scalar(message.repeated_scalar.val.add(), '_ngraph_iter_sentinel_')
    elif value is None:
        message.scalar.null_val = True
    else:
        # raise unhandled_scalar_value(value)
        print('skipped unhandled_scalar_value {} in serde.'.format(value))


def op_to_protobuf(op):
    """
    Converts all attributes of an op into protobuf values and returns it. Skips over the
    properties of `args`, `ops`, `control_deps`, and `forward` since those are added as
    edges separately.
    """
    pb_op = ops_pb.Op(name=op.name, op_type=op.__class__.__name__)
    if hasattr(op, 'dtype'):
        pb_op.dtype = dtype_to_protobuf(op.dtype)
    pb_op.uuid.uuid = op.uuid.bytes

    # Hoist metadata into the general purpose attrs dict with namespacing
    for key in op.metadata:
        assign_op_attr(pb_op.attrs['_ngraph_metadata_' + key], op.metadata[key])

    if hasattr(op, '_ngraph_ser_handle'):
        pb_op.attrs['_ngraph_ser_handle'].scalar.bool_val = True
        # cleanup our sentinel value
        del op._ngraph_ser_handle
    # TODO(jknight) This is a hack
    # We run the valfun closure and store the result rather than serialize the closure
    # We could instead serialize the closure, but this has its own issues (see Keras utils and
    # issus tracker for gory details)
    if hasattr(op, 'valfun'):
        pb_op.attrs['valfun_value'].tensor.CopyFrom(
            tensor_to_protobuf(op.valfun(op.tensor_description())))

    # These are handled above
    ignored_keys = {'valfun', 'uuid', 'dtype', 'metadata'}
    remaining_keys = set(op.__dict__.keys()).difference(ignored_keys)

    for key in remaining_keys:
        if not key.startswith('_is_') and key not in EXCEPTION_ATTRIBUTES and key.startswith('_'):
            continue
        val = getattr(op, key)
        if isinstance(val, Op) or \
            (isinstance(val, (list, set, tuple)) and
             len(val) > 0 and
             all(map(lambda x: isinstance(x, Op), val))):
            # These will be handled in `add_edges`
            continue
        assign_op_attr(pb_op.attrs[key], getattr(op, key))
    return pb_op


def add_edges(pb_edges, pb_ops, op):
    """ Adds the edges present in `op` to `pb_edges` and `pb_ops` lists. """

    def add_edge(from_op, to_op, edge_type):
        edge = ops_pb.Edge()
        edge.uuid.uuid = uuid.uuid4().bytes
        edge.from_uuid.uuid = from_op.uuid.bytes
        edge.to_uuid.uuid = to_op.uuid.bytes
        edge.edge_type = edge_type
        pb_edges.append(edge)
        return edge

    if op._forward is not None:
        forward_edge = add_edge(op, op.forward, ops_pb.Edge.CONTAINER)
        forward_edge.attrs['_ngraph_forward'].scalar.bool_val = True
    if hasattr(op, '_args'):
        for arg in op._args:
            add_edge(arg, op, ops_pb.Edge.DATA)
    if hasattr(op, '_ops'):
        for arg in op._ops:
            add_edge(op, arg, ops_pb.Edge.CONTAINER)
    if hasattr(op, '_control_deps'):
        for arg in op._control_deps:
            add_edge(op, arg, ops_pb.Edge.CONTROL)

    # Now iterate through remaining keys of this op's __dict__ and any that reference
    # other Ops we make edges that we can deserialize as Op attributes later
    remaining_keys = set(op.__dict__.keys())
    for key in remaining_keys:
        if not key.startswith('_is_') and key not in EXCEPTION_ATTRIBUTES and key.startswith('_'):
            continue
        val = getattr(op, key)
        if isinstance(val, Op):
            edge = add_edge(op, val, ops_pb.Edge.OTHER)
            edge.attrs['_ngraph_attribute'].scalar.string_val = key
        if isinstance(val, (list, tuple, set)):
            for item in val:
                if isinstance(item, Op):
                    edge = add_edge(op, item, ops_pb.Edge.OTHER)
                    edge.attrs['_ngraph_list_attribute'].scalar.string_val = key
            # TODO(jknight): assert that ALL values of this list are op references


def _serialize_graph(ops):
    """
    Serializes a graph and returns the actual protobuf python object (rather than serialized
    byte string as done by `serialize_graph`).
    """
    assert isinstance(ops, Iterable), "Ops passed into `serialize_graph` must be an iterable"
    ops = Op.all_op_references(ops)
    pb_ops = []
    pb_edges = []
    for op in ops:
        pb_ops.append(op_to_protobuf(op))
        add_edges(pb_edges, pb_ops, op)

    graph_def = ops_pb.GraphDef()
    for edge in pb_edges:
        temp = graph_def.edges.add()
        temp.CopyFrom(edge)
    for op in pb_ops:
        temp = graph_def.ops.add()
        temp.CopyFrom(op)
    return graph_def


def serialize_graph(ops, only_return_handle_ops=False):
    """
    Dumps ngraph graph to serialized protobuf byte string

    Params:
      only_return_handle_ops <bool>: If false, this will return ALL ops upon deserialization. If
          true, then only the ops passed in to be serialized will be returned upon deserialization
          (with links to upstream ops intact).
    """
    if only_return_handle_ops:
        for op in ops:
            op._ngraph_ser_handle = True
    return _serialize_graph(ops).SerializeToString()


##################
# DESERIALIZATION
##################


def pb_to_dtype(pb_dtype):
    return np.dtype(getattr(np, ops_pb.DTYPE.Name(pb_dtype).lower()))


def pb_to_dict(map_val):
    return {key: protobuf_scalar_to_python(map_val[key]) for key in map_val.keys()
            if key != '_ngraph_map_sentinel_'}


def pb_to_tensor(pb_tensor):
    np_dtype = pb_to_dtype(pb_tensor.dtype)
    if np_dtype == np.float64:
        data = pb_tensor.double_data
    elif np_dtype == np.float32 or np_dtype == np.float16:
        data = pb_tensor.float_data
    elif np_dtype == np.int64:
        data = pb_tensor.int_data
    if len(pb_tensor.shape) == 0:
        return np_dtype.type(data[0])
    else:
        return np.array(data, dtype=np_dtype).reshape(pb_tensor.shape)


def protobuf_scalar_to_python(val):
    assert isinstance(val, ops_pb.Scalar)
    scalar_key = val.WhichOneof('value')
    if scalar_key == 'uuid_val':
        raise ValueError("During deserialization, no attributes should reference UUIDs.")
    elif scalar_key == 'map_val':
        return pb_to_dict(val.map_val.map)
    elif scalar_key == 'null_val':
        return None
    elif scalar_key == 'slice_val':
        val = val.slice_val
        return slice(val.start.value if val.HasField('start') else None,
                     val.stop.value if val.HasField('stop') else None,
                     val.step.value if val.HasField('step') else None)
    elif scalar_key == 'dtype_val':
        return pb_to_dtype(val.dtype_val)
    return getattr(val, scalar_key)


def protobuf_to_axes(msg):
    if msg.uuid.uuid in GLOBAL_AXIS_REGISTRY:
        return GLOBAL_AXIS_REGISTRY[msg.uuid.uuid]
    axes = ngraph.make_axes(list(map(pb_to_axis, msg.axes)))
    axes.uuid = uuid.UUID(bytes=msg.uuid.uuid)
    axes.name = msg.name
    GLOBAL_AXIS_REGISTRY[msg.uuid.uuid] = axes
    return axes


def protobuf_attr_to_python(val):
    if val.HasField('scalar'):
        return protobuf_scalar_to_python(val.scalar)
    if val.HasField('tensor'):
        return pb_to_tensor(val.tensor)
    elif val.HasField('repeated_scalar'):
        if len(val.repeated_scalar.val) == 1 and \
                val.repeated_scalar.val[0].string_val == '_ngraph_iter_sentinel_':
            return ()
        else:
            return list(map(protobuf_scalar_to_python, val.repeated_scalar.val))
    elif val.HasField('axes'):
        return protobuf_to_axes(val.axes)
    elif val.HasField('axis'):
        return pb_to_axis(val.axis)
    elif str(val) == '':
        # hetr only, for shared queues
        # shared queues skipped serialization, so val.__str__ will be '' here
        pass
    else:
        raise ValueError("Cannot convert {} to python attribute value".format(val))


def pb_to_axis(msg):

    if msg.uuid.uuid in GLOBAL_AXIS_REGISTRY:  # Already deserialized
        return GLOBAL_AXIS_REGISTRY[msg.uuid.uuid]
    elif msg.HasField('flattened_axes'):  # FlattenedAxis
        axes = protobuf_to_axes(msg.flattened_axes)
        axis = FlattenedAxis(axes)
    else:
        axis = Axis(name=msg.name,
                    length=msg.length)

    axis.uuid = uuid.UUID(bytes=msg.uuid.uuid)
    GLOBAL_AXIS_REGISTRY[axis.uuid.bytes] = axis
    return axis


def get_ngraph_op_cls(op_type):
    """ Walk over python modules in ngraph.op_graph and look for op_type class. """
    for importer, modname, ispkg in pkgutil.iter_modules(ngraph.op_graph.__path__):
        imported_mod = importlib.import_module('ngraph.op_graph.' + modname)
        if hasattr(imported_mod, op_type):
            return getattr(imported_mod, op_type)
    raise ValueError("Cannot find op_type of {} in any ngraph.op_graph modules.".format(op_type))


def protobuf_to_op(pb_op):
    """
    This will convert a protobuf Op object into its corresponding Python object. But this cannot
    setup links to other ops (such as args, control_deps) since those ops may not
    exist yet.
    We have to wait until all ops are created before connecting them back up together in a second
    pass, so args, etc will be uninitialized.
    """
    cls = get_ngraph_op_cls(pb_op.op_type)

    # Skip the class constructor but we'll use the generic op constructor because it sets a lot of
    # helpful defaults
    py_op = cls.__new__(cls)
    op_graph.Op.__init__(py_op)
    py_op.name = pb_op.name

    if 'valfun_value' in pb_op.attrs:
        valfun_value = pb_to_tensor(pb_op.attrs['valfun_value'].tensor)
        py_op.valfun = lambda x: valfun_value

    # op.uuid
    py_op.uuid = uuid.UUID(bytes=pb_op.uuid.uuid)

    # op.metadata and remaining keys
    ignored_keys = {'valfun_value', 'dtype', 'metadata'}
    remaining_keys = set(pb_op.attrs.keys()).difference(ignored_keys)
    for key in remaining_keys:
        if key == '_ngraph_ser_handle':
            py_op._ngraph_ser_handle = True
        if key.startswith('_ngraph_metadata_'):
            value = pb_op.attrs[key]
            py_op.metadata[key[17:]] = protobuf_attr_to_python(value)
        elif not key.startswith('_is_') and key not in EXCEPTION_ATTRIBUTES and \
                key.startswith('_'):
            continue
        else:
            value = pb_op.attrs[key]
            setattr(py_op, key, protobuf_attr_to_python(value))
    return py_op


def _deserialize_graph(graph_pb):
    """
    Will deserialize a graph and return the list of all ops in that graph. Does not bother
    filtering down to only the original set of ops the user passed in for serialization
    (if that's what the user desired upon serializing with the serialization
    only_return_handle_ops parameter).
    """
    # For safety we clear this registry
    GLOBAL_AXIS_REGISTRY.clear()

    ops = list(map(protobuf_to_op, graph_pb.ops))
    uuid_lookup = {op.uuid.bytes: op for op in ops}
    for edge in graph_pb.edges:
        head_op = uuid_lookup[edge.from_uuid.uuid]
        tail_op = uuid_lookup[edge.to_uuid.uuid]
        if edge.edge_type == ops_pb.Edge.DATA:  # args
            tail_op._args = tail_op._args + (head_op,)
        elif edge.edge_type == ops_pb.Edge.CONTROL:  # control_deps
            head_op._control_deps.add(tail_op)
        elif edge.edge_type == ops_pb.Edge.CONTAINER:
            if '_ngraph_forward' in edge.attrs:  # forward
                head_op._forward = tail_op
            else:  # ops
                if not hasattr(head_op, '_ops'):
                    head_op._ops = []
                head_op._ops.append(tail_op)
        elif edge.edge_type == ops_pb.Edge.OTHER:
            if '_ngraph_attribute' in edge.attrs:
                setattr(head_op, edge.attrs['_ngraph_attribute'].scalar.string_val, tail_op)
            elif '_ngraph_list_attribute' in edge.attrs:
                key = edge.attrs['_ngraph_list_attribute'].scalar.string_val
                # import pdb; pdb.set_trace()
                if hasattr(head_op, key):
                    getattr(head_op, key).add(tail_op)
                else:
                    setattr(head_op, key, OrderedSet([tail_op]))
        else:
            raise ValueError("Edge not mapped to op: {}".format(edge))

    # This must come after tensor has been set which occurs after edges
    # op.dtype
    for py_op, pb_op in zip(ops, graph_pb.ops):
        py_op.dtype = pb_to_dtype(pb_op.dtype)

    # Done with this and don't want it to bleed to subsequent serializations
    GLOBAL_AXIS_REGISTRY.clear()

    # Assemble list of nodes to return depending on if the user wanted to
    # return all ops or only those that were originally serialized (and
    # implicitly those upstream)
    final_ops = []
    for op in ops:
        if hasattr(op, '_ngraph_ser_handle'):
            del op._ngraph_ser_handle
            final_ops.append(op)
    if len(final_ops) > 0:
        return final_ops
    else:
        return ops


def deserialize_graph(graph_msg):
    """
    Given a serialized protobuf `GraphDef` bytestring, this will deserialize it and return
    the Ops of the graph.
    """
    return _deserialize_graph(ops_pb.GraphDef.FromString(graph_msg))
