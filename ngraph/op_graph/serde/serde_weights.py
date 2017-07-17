import uuid

import six
from zipfile import ZipFile
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse

import ngraph as ng
from ngraph.op_graph.serde import ops_pb2
from ngraph.op_graph.serde.serde import dtype_to_protobuf, data_to_tensor

MANIFEST_FILENAME = '__MANIFEST__'


def write_raw_np(value, f):
    f.write(value.tostring())


def json_dumps_manifest(values):
    """
    Arguments:
        values: {str: np.array}
    """
    manifest = ops_pb2.TensorManifest()
    for uuid_val, a in values.items():
        pair = manifest.pairs.add()
        if isinstance(uuid_val, six.binary_type):
            pair.uuid.uuid = uuid_val
        else:
            pair.uuid.uuid = uuid_val.encode()
        pair.info.dtype = dtype_to_protobuf(a.dtype)
        pair.info.shape.extend(a.shape)
    return MessageToJson(manifest)


def json_loads_manifest(js):
    """
    Returns: {str: tensor}
    """
    manifest = ops_pb2.TensorManifest()
    Parse(js, manifest)

    return {t.uuid.uuid: t.info for t in manifest.pairs}


def write_np_values(values, f):
    """
    Arguments:
        values: {str: np.array}
        f: filename or filelike object
    """
    with ZipFile(f, 'w') as zf:
        for k, v in values.items():
            # Need to do this because Python zipfile has some odd support for filenames:
            # http://bugs.python.org/issue24110
            if len(k) == 16 and isinstance(k, six.binary_type):  # valid UUID bytes
                zf.writestr(str(uuid.UUID(bytes=k)), v.tostring())
            else:
                zf.writestr(six.u(k), v.tostring())

        zf.writestr(MANIFEST_FILENAME, json_dumps_manifest(values))


def read_np_values(f):
    """
    given a filename or filelike ``f``, read weights out and return {uuid: np.array}
    """
    values = {}

    with ZipFile(f, 'r') as zf:
        manifest = json_loads_manifest(zf.read(MANIFEST_FILENAME))

        for filename in zf.namelist():
            if filename == MANIFEST_FILENAME:
                continue

            # Need to do this because Python zipfile has some odd support for filenames:
            # http://bugs.python.org/issue24110
            try:
                uuid_val = uuid.UUID(filename)
                values[uuid_val.bytes] = data_to_tensor(
                    zf.read(filename), manifest[uuid_val.bytes]
                )
            except ValueError:
                values[filename] = data_to_tensor(
                    zf.read(filename), manifest[six.b(filename)]
                )

    return values


##################
# extract values out of and set value into ops by uuid
##################

def extract_op(transformer, op):
    """
    Returns a numpy array with the value of the tensor `op`
    """
    return transformer.computation(op)()


def extract_ops(transformer, ops):
    """
    Returns a {uuid: np.array} containing a map from each op's uuid to its
    value, for all ops in `ops`.
    """
    return {op.uuid.bytes: extract_op(transformer, op) for op in ops}


def set_op_value(transformer, op, value):
    """
    Given an op and a numpy array, set the op's value to the numpy array
    """
    transformer.computation(ng.AssignOp(op, value))()


def set_op_values(transformer, ops, op_values):
    """
    Given a set of ops and an op_value map (as from extract_ops), set the value
    of each of the ops in `ops`.
    """
    for op in ops:
        set_op_value(transformer, op, op_values[op.uuid.bytes])


# compose extraction and serialization

def serialize_weights(transformer, ops, f):
    """
    Serialize the weights of a nervana graph object to a given filename of file-like object.
    Arguments:
        transformer: <Transformer> The transformer that maintains the values of `ops` that
            you want to extract.
        ops: <Op>, the terminal op of the graph that you want to serialize the weights from.
        f: <string or file-like>: The name or file object you want to write the weights into.
    """
    return write_np_values(extract_ops(transformer, ops), f)


def deserialize_weights(transformer, ops, f):
    """
    De-serialize the weights of a nervana graph object from a given filename or file-like object.
    Arguments:
        transformer: <Transformer> The transformer that maintains the values of `ops` that
            you want to extract.
        ops: <Op>, the terminal op of the graph that you want to serialize the weights from.
            NB: These ops must be deserialized from disk or be the same ops (in memory) that
            were used for the original weight serialization. If not, then the UUIDs will differ
            between newly generated ops (even from the same topology) and the UUIDs of the
            previous ops that the weights are linked to.
        f: <string or file-like>: The name or file object you want to write the weights into.
    """
    return set_op_values(transformer, ops, read_np_values(f))
