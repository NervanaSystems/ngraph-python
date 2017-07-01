import os
import subprocess
import tempfile

from collections import Iterable

from ngraph.op_graph.serde.serde import _serialize_graph

# Tensorflow is (should be anyways) an optional dependency
# for ngraph, and we only want to fail if the functions below
# are invoked without TF installed.
try:
    import tensorflow as tf
    from tensorflow.core.framework import graph_pb2
    from tensorflow.core.util import event_pb2
    from tensorflow.python import pywrap_tensorflow
    from tensorflow.python.util import compat
    from tensorflow.python.framework import errors
    TF_IMPORT_SUCCESS = True
except ImportError:
    TF_IMPORT_SUCCESS = False


def ngraph_to_tf_graph_def(graph):
    """
    Given an ngraph graph, convert it to a TensorFlow `GraphDef` protobuf object in memory.
    Arguments:
        graph: Op or list of ops
    Returns:
        A Tensorflow `tensorflow.core.framework.graph_pb2.GraphDef` structure.
    """
    if not TF_IMPORT_SUCCESS:
        raise ImportError("Tensorflow is not installed, yet it is required ",
                          "to export Nervana Graph IR to TF GraphDef")

    if not isinstance(graph, Iterable):
        graph = [graph]
    ng_graph_def = _serialize_graph(graph)

    op_names = set()
    op_uuid_map = dict()
    tf_graph_def = graph_pb2.GraphDef()
    for op in ng_graph_def.ops:
        node_def = tf_graph_def.node.add()
        if '_ngraph_metadata_neon_layer' in op.attrs:
            node_def.name = op.attrs['_ngraph_metadata_neon_layer'].scalar.string_val + \
                '/' + op.name
        else:
            node_def.name = op.name
        node_def.op = op.op_type
        if op.name in op_names:
            raise ValueError("Op with name {} exists in duplicate".format(op.name))
        op_names.add(op.name)
        op_uuid_map[op.uuid.uuid] = node_def

    for edge in ng_graph_def.edges:
        from_op = op_uuid_map[edge.from_uuid.uuid]
        to_op = op_uuid_map[edge.to_uuid.uuid]
        to_op.input.append(from_op.name)

    return tf_graph_def


def ngraph_to_tensorboard(graph, start_tensorboard=False):
    """
    Given an ngraph `graph` in the form of an op or list of ops, translate this into a TensorFlow
    compatible format and launch Tensorboard to visualize it.
    Arguments:
        graph: Op or list of Ops - Graph to visualize.
        start_tensorboard: Bool - Whether to launch tensorboard with the right `--log_dir`
            argument.  If false, then the User is responsible for taking the printed out file
            path and using that as an argument to tensorboard.
    Returns:
        Path to generated TensorFlow `Record` format (with a length delimited and checksummed
            append type structure) where each entry is a binary encoded `Event` protobuf record.
    """
    if not TF_IMPORT_SUCCESS:
        raise ImportError("Tensorflow is not installed, yet it is required ",
                          "to export Nervana Graph IR to Tensorboard")
    dname = tempfile.mkdtemp()
    fname = tempfile.mktemp(dir=dname, prefix='events.out.tfevents.ngraph.')

    tf_graph_def = ngraph_to_tf_graph_def(graph)
    with errors.raise_exception_on_not_ok_status() as status:
        writer = pywrap_tensorflow.PyRecordWriter_New(compat.as_bytes(fname),
                                                      compat.as_bytes(''),
                                                      status)

    ev = event_pb2.Event()
    ev.graph_def = tf_graph_def.SerializeToString()
    writer.WriteRecord(ev.SerializeToString())

    # Upstream API change to writer.Close coming in TF 1.3
    # https://github.com/tensorflow/tensorflow/commit/abae4305a6208bde6072d257a6ce734a8d369089#diff-39e499ff7be73304a41e23efd8432f40L45
    if tf.__version__ >= '1.3':
        with errors.raise_exception_on_not_ok_status() as status:
            writer.Close(status)
    else:
        writer.Close()

    print(fname)

    if start_tensorboard:
        subprocess.check_call(['tensorboard', '--logdir', os.path.dirname(fname)])
    return fname
