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
import numpy as np
import ngraph as ng
from future.utils import viewitems
import six
from ngraph.frontends.neon import ax
import collections


class RollingWindowIterator(object):

    def __init__(self, data_array, batch_size, seq_len, total_iterations=None):
        """
        Given an input sequence, generates overlapping windows of samples

        Input: numpy array
            data_array : Numpy array of shape (N,D).
                                N is length of sequence
                                D is input feature dimension
        Output of each iteration: Dictionary of of input and output samples
            samples['X'] has size (batch_size, seq_len, D) 
            samples['y'] has size (batch_size, 1, D)

        Example:
            data_array is a numpy array with shape (N,1): [a1, a2, ..., aN]
            Each generated sample will be an input sequence / output single point pairs such as:
                Sample0: Input:  [a1, a2, ..., a(seq_len)], Output: a(seq_len + 1)
                Sample1: Input:  [a2, a3, ..., a(seq_len+1)], Output: a(seq_len +2)
                        ...
            Each iteration will return batch_size number of samples

        seq_len: Width of the rolling window requested
        batch_size: how many samples to return for each iteration
        total_iterations: number of batches to retrieve from the sequence (roll over if necessary)
                         If set to None, will rotate through the whole sequence only once 
        """

        # Treat singletons like list so that iteration follows same syntax
        self.batch_size = batch_size

        # Get the total length of the sequence
        self.ndata = len(self.data_array)

        if self.ndata < self.batch_size:
            raise ValueError('Number of examples is smaller than the batch size')

        self.start = 0
        self.index = 0

        self.total_iterations = self.nbatches if total_iterations is None else total_iterations

    @property
    def nbatches(self):
        """
        Return the number of minibatches in this dataset.
        """
        return -((self.start - self.ndata) // self.batch_size)

    def make_placeholders(self, include_iteration=False):
        placeholders = {}
        ax.N.length = self.batch_size
        for k, axnm in self.axis_names.items():
            p_axes = ng.make_axes([ax.N])
            for i, sz in enumerate(self.data_arrays[k].shape[1:], 1):
                name = axnm[i] if axnm else None
                p_axes += ng.make_axis(length=sz, name=name)
            placeholders[k] = ng.placeholder(p_axes)
        if include_iteration:
            placeholders['iteration'] = ng.placeholder(axes=())
        return placeholders

    def reset(self):
        """
        Resets the starting index of this dataset to zero. Useful for calling
        repeated evaluations on the dataset without having to wrap around
        the last uneven minibatch. Not necessary when data is divisible by batch size
        """
        self.start = 0
        self.index = 0

    def __iter__(self):
        """
        Returns a new minibatch of data with each call.

        Yields:
            dictionary: The next minibatch which includes both features and labels.
                samples['X']: Features, with shape (batch_size, seq_len, feature_dim) 
                samples['y']: Labels, with shape (batch_size, 1, feature_dim)
        """
        while self.index < self.total_iterations:
            strt_idx = (self.start + self.index * self.batch_size) 
            end_idx = strt_idx + self.seq_len + self.batch_size
            idcs = np.arange(strt_idx, end_idx) % self.ndata
            self.index += 1

            samples['iteration'] = self.index
            yield samples
