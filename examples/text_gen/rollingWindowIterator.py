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
from __future__ import division, print_function
import numpy as np


class RollingWindowIterator(object):

    def __init__(self, data_array, batch_size, seq_len, total_iterations=None, stride=1):
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
                Sample0: Input:  [a1, a2, ..., a(seq_len)]
                         Output: a(seq_len + 1)
                Sample1: Input:  [a(stride +1), a(stride+2), ..., a(stride+seq_len)]
                         Output: a(stride+seq_len +1)
                        ...
            Each iteration will return batch_size number of samples

        If stride = 1, the window will shift by one
            Hence Sample0 and Sample1 will have (seq_len - 1) elements that are the same
        If stride = seq_len, Sample0 and Sample1 will have no overlapping elements

        seq_len: Width of the rolling window requested
        batch_size: how many samples to return for each iteration
        total_iterations: number of batches to retrieve from the sequence (roll over if necessary)
                         If set to None, will rotate through the whole sequence only once
        """

        self.data_array = data_array
        self.seq_len = seq_len
        if (len(data_array.shape) == 1):
            self.data_array = self.data_array[:, np.newaxis]

        if (stride is None):
            self.stride = 1
        else:
            self.stride = stride
        self.feature_dim = self.data_array.shape[1]

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
        return -((self.start - self.ndata) // self.batch_size // self.stride)

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
        samples = {'X': np.zeros((self.batch_size, self.seq_len, self.feature_dim)),
                   'y': np.zeros((self.batch_size, 1, self.feature_dim))}
        stride = self.stride
        while self.index < self.total_iterations:
            strt_idx = (self.start + self.index * self.batch_size * stride)
            end_idx = strt_idx + self.seq_len
            self.index += 1
            sample_id = 0
            for batch_idx in range(self.batch_size):
                idcs = np.arange(strt_idx + (batch_idx * stride),
                                 end_idx + (batch_idx * stride)) % self.ndata
                samples['X'][sample_id] = self.data_array[idcs]
                samples['y'][sample_id] = self.data_array[(idcs[-1] + 1) % self.ndata]
                sample_id += 1
            samples['iteration'] = self.index
            yield samples
