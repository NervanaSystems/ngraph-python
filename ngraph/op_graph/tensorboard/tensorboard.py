from __future__ import absolute_import

import os
import subprocess
import tempfile
import logging
import time
import struct
import socket
import datetime as dt

from collections import Iterable

import numpy as np
from ngraph.op_graph.serde.serde import _serialize_graph
from ngraph.op_graph.tensorboard.tfrecord import masked_crc32c
from ngraph.op_graph.tensorboard import summary

# Tensorflow is (should be anyways) an optional dependency
# for ngraph, and we only want to fail if the functions below
# are invoked without TF installed.
try:
    from tensorflow.core.framework import graph_pb2
    from tensorflow.core.util import event_pb2
    from tensorflow.core.framework import summary_pb2
    TF_IMPORT_SUCCESS = True
except ImportError:
    TF_IMPORT_SUCCESS = False


logger = logging.getLogger(__name__)


class Tensorboard(object):

    def __init__(self, logdir, run=None):
        """
        Creates an interface for logging ngraph data to tensorboard
        
        Arguments:
            logdir (str): Path to tensorboard logdir
            run (str, optional): Name of the current run. If one is not provided, the run name 
                                 will be generated from the date and time.
            
        Notes:
            1. Tensorboard must be started separately from the terminal using `tensorboard --logdir 
               <logdir>`
            2. In tensorboard, "runs" denote individual experiments that can be browsed 
               simultaneously. In this way, it's useful to use the same logdir for multiple 
               experiments that are conceptually related. 
            
        """

        self.logdir = logdir
        self.run = run
        self._record_file = None

        if not os.path.isdir(logdir):
            os.makedirs(logdir)
        if self.run is not None:
            self.add_run(self.run)

    def add_run(self, run=None):
        """
        Create a new run and start using it.
        
        Arguments:
            run (str, optional): Name of the new run. If one is not provided, it will be 
                                 generated from the current date and time. If the run already 
                                 exists, it will be reused.
        """


        def get_events_filename():
            return ".".join(["events.ngraph.out.tfevents",
                             str(time.time())[:10],
                             socket.gethostname()])

        if run is None:
            run = dt.datetime.strftime(dt.datetime.now(), "%y%m%dT%H%M%S")
        self.run = run

        directory = os.path.join(self.logdir, self.run)
        if os.path.isdir(directory):
            files = [fname for fname in os.listdir(directory) if "out.tfevents" in fname]
            if len(files) > 0:
                record_file = os.path.join(directory, files[0])
            else:
                record_file = os.path.join(directory, get_events_filename())
        else:
            os.makedirs(directory)
            record_file = os.path.join(directory, get_events_filename())

        self._record_file = record_file

    def add_scalar(self, name, scalar, step=None):
        """
        Add a scalar to the current tensorboard run
        
        Arguments:
            name (str): Display name within tensorboard
            scalar (int, float): Scalar value to be logged 
            step (int, optional): Step in the series. Optional, but should usually be provided. 
        """

        summ = summary.scalar(name, scalar)
        self._write_event(create_event(summary=summ, step=step))

    def add_image(self, name, image, step=None):
        """
        Add an image to the current tensorboard run

        Arguments:
            name (str): Display name within tensorboard
            image (ndarray): 3-D array containing the image to be logged. It should be 
                             formatted as Height by Width by Channels, where 1 channel is 
                             greyscale, 3 channels is RGB, and 4 channels is RGBA.
            step (int, optional): Step in the series
        """

        summ = summary.image(name, image)
        self._write_event(create_event(summary=summ, step=step))

    def add_histogram(self, name, sequence, step=None):
        """
        Add a histogram to the current tensorboard run

        Arguments:
            name (str): Display name within tensorboard
            sequence (ndarray, Iterable): A sequence on which to compute a histogram of values
            step (int, optional): Step in the series. Optional, but should usually be provided. 
        """

        if isinstance(sequence, np.ndarray):
            sequence = np.ravel(sequence)
        summ = summary.histogram(name, sequence)
        self._write_event(create_event(summary=summ, step=step))

    def add_graph(self, ops):
        """
        Add a graph to the current tensorboard run
        Arguments:
            ops (Op, Iterable): Ops to serialize as a TensorFlow Graph 
        """
        graph_def = ngraph_to_tf_graph_def(ops)
        self._write_event(create_event(graph_def=graph_def))

    def add_audio(self, name, audio, sample_rate, step=None):
        """
        Add an audio clip to the current tensorboard run

        Arguments:
            name (str): Display name within tensorboard
            audio (ndarray): 2-D array containing the audio clip to be logged. It should be 
                             formatted as Frames by Channels.
            sample_rate (float): Sample rate of audio in Hertz
            step (int, optional): Step in the series            
        """

        summ = summary.audio(name, audio, sample_rate)
        self._write_event(create_event(summary=summ, step=step))

    def _write_event(self, event):
        """ Writes an event to the current TensorFlow record file"""
        if self.run is None:
            self.add_run()
        with RecordWriter(self._record_file, "ab") as fh:
            fh.write(event)


def ngraph_to_tf_graph_def(graph):
    """
    Given an ngraph graph, convert it to a TensorFlow `GraphDef` protobuf object in memory.

    Arguments:
        graph (Op, Iterable): Ops to serialize as a TensorFlow Graph

    Returns:
        A Tensorflow `tensorflow.core.framework.graph_pb2.GraphDef` structure.

    References:
        Tensorflow graphdef proto:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto
        Tensorflow nodedef proto:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto
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
        node_def.name = op.name
        node_def.op = op.op_type
        node_def.attr['axes'].s = str(op.attrs['axes'].scalar.string_val)
        for key, value in op.attrs.items():
            if key.startswith("_ngraph_metadata_"):
                key = key.replace("_ngraph_metadata_", "")
                value_type = value.scalar.WhichOneof("value")
                if value_type == "string_val":
                    node_def.attr[key].s = value.scalar.string_val
                elif value_type == "bool_val":
                    node_def.attr[key].b = value.scalar.bool_val
                elif value_type == "double_val":
                    node_def.attr[key].f = value.scalar.double_val
                elif value_type == "int_val":
                    node_def.attr[key].i = value.scalar.int_val
                else:
                    # TODO: Could also capture slice and dtype
                    pass

        if op.name in op_names:
            raise ValueError("Op with name {} exists in duplicate".format(op.name))
        op_names.add(op.name)
        op_uuid_map[op.uuid.uuid] = node_def

    for edge in ng_graph_def.edges:
        from_op = op_uuid_map[edge.from_uuid.uuid]
        to_op = op_uuid_map[edge.to_uuid.uuid]
        to_op.input.append(from_op.name)

    return tf_graph_def


def serialize_protobuf(pb):

    if not isinstance(pb, bytes):
        if not hasattr(pb, "SerializeToString"):
            raise TypeError("pb must be a bytestring or a protobuf object, not {}".format(type(pb)))
        pb = pb.SerializeToString()

    return pb


def deserialize_protobuf(pb, pb_type):

    if isinstance(pb, bytes):
        pb = pb_type.FromString(pb)

    if not isinstance(pb, pb_type):
        raise TypeError("pb must be a bytestring or a protobuf of type {}, not {}".format(pb_type,
                                                                                          type(pb)))
    return pb


def event_to_record(event):
    """
    Convert an event protobuf to a tfrecord
    Args:
        event: 

    Returns:

    """

    event_str = serialize_protobuf(event)
    header = struct.pack('Q', len(event_str))
    record = [header,
              struct.pack('I', masked_crc32c(header)),
              event_str,
              struct.pack('I', masked_crc32c(event_str))]

    return b"".join(record)


class RecordWriter(object):
    def __init__(self, f, mode='wb'):
        """
        Create a tfrecord writer
        Arguments:
            f (str): Path to record file 
            mode (str): Mode to open file (must be one of 'wb' or 'ab') 
        """
        if mode not in ('wb', 'ab'):
            raise ValueError("mode must be one of 'wb' or 'ab', not {}".format(mode))

        self._f = f
        self._mode = mode
        self._file_obj = None
        self._written = 0

    def __enter__(self):
        self._file_obj = open(self._f, self._mode)
        return self

    def __exit__(self, *args, **kwargs):
        self._file_obj.close()

    def write(self, event):
        if self._written == 0:
            self._file_obj.write(event_to_record(create_event()))
        self._file_obj.write(event_to_record(event))
        self._file_obj.flush()
        self._written += 1


def create_event(summary=None, graph_def=None, wall_time=None, step=None, **kwargs):

    if summary is not None:
        event = event_pb2.Event(summary=deserialize_protobuf(summary, summary_pb2.Summary))
    elif graph_def is not None:
        event = event_pb2.Event(graph_def=serialize_protobuf(graph_def))
    else:
        event = event_pb2.Event()

    if wall_time is None:
        wall_time = time.time()
    event.wall_time = wall_time

    if step is not None:
        event.step = int(step)

    return event