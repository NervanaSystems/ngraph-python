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

import os
import logging
import time
import socket
import datetime as dt

import numpy as np
from ngraph.op_graph.tensorboard import summary
from ngraph.op_graph.tensorboard.graph_def import ngraph_to_tf_graph_def
from ngraph.op_graph.tensorboard.tfrecord import RecordWriter, create_event


logger = logging.getLogger(__name__)


class TensorBoard(object):

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
