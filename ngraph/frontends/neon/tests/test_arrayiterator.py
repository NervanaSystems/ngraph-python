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
import pytest
import numpy as np
from ngraph.frontends.neon import SequentialArrayIterator


@pytest.fixture(scope='module',
                params=[
                    np.arange(10000),
                    np.arange(20000).reshape(10000, 2),
                    np.arange(10001)],
                ids=[
                    'even_seq',
                    'multidim_seq',
                    'odd_seq'])
def input_seq(request):
    return request.param


@pytest.fixture(scope='module',
                params=[8, 32])
def batch_size(request):
    return request.param


@pytest.fixture(scope='module',
                params=[16, 32])
def seq_len(request):
    return request.param


def test_nowindow(input_seq, batch_size, seq_len):
    """
    This test checks no-shuffle option, with non-overlapping windows
    Goes over the dataset just once
    """
    # Create the iterator
    data_array = {'X': np.copy(input_seq),
                  'y': np.roll(input_seq, axis=0, shift=-1)}
    iterations = None
    it_array = SequentialArrayIterator(data_arrays=data_array, time_steps=seq_len,
                                       batch_size=batch_size, tgt_key='y', shuffle=False,
                                       total_iterations=iterations)

    # Cut off the last samples of the input sequence that don't fit in a batch
    no_samples = seq_len * batch_size * (len(input_seq) // seq_len // batch_size)
    target_seq = input_seq[1:no_samples + 1]
    input_seq = input_seq[:no_samples]

    for idx, iter_val in enumerate(it_array):
        # Start of the array needs to be time_steps * idx * batch_size
        start_idx = seq_len * idx * batch_size
        idcs = np.arange(start_idx, start_idx + seq_len * batch_size) % input_seq.shape[0]

        # Reshape the input sequence into batches
        reshape_dims = (batch_size, seq_len)
        if len(input_seq.shape) > 1:
            reshape_dims = (batch_size, seq_len, input_seq.shape[1])
        # expected_x will have contigous non-overlapping samples
        expected_x = input_seq[idcs].reshape(reshape_dims)
        expected_y = target_seq[idcs].reshape(reshape_dims)

        # We are not shuffling, consecutive samples are contiguous
        # They will also be non-overlapping
        assert np.array_equal(expected_x, iter_val['X'])
        assert np.array_equal(expected_y, iter_val['y'])


def test_shuffle(input_seq, batch_size, seq_len):
    """
    This test checks shuffle option, with non-overlapping windows
    Goes over the dataset just once
    """
    def sample2char(sample):
        # Converts a numpy array into a single string
        # sample is a numpy array
        # We convert to string so that the data can be hashable to be used in a python set
        str_array = ''.join(['%1.2f' % i for i in sample.flatten()])
        return str_array
    # create iterator
    data_array = {'X': np.copy(input_seq),
                  'y': np.roll(input_seq, axis=0, shift=-1)}
    it_array = SequentialArrayIterator(data_arrays=data_array, time_steps=seq_len,
                                       batch_size=batch_size, tgt_key='y', shuffle=True,
                                       total_iterations=None)

    # Cut off the last samples of the input sequence that don't fit in a batch
    no_batches = len(input_seq) // seq_len // batch_size
    no_samples = batch_size * no_batches
    used_steps = seq_len * no_samples
    input_seq = input_seq[:used_steps]

    # Reshape the input sequence into batches
    reshape_dims = (no_samples, seq_len)
    if len(input_seq.shape) > 1:
        reshape_dims = (no_samples, seq_len, input_seq.shape[1])
    expected_x = input_seq.reshape(reshape_dims)
    # Convert each sample into text and build a set
    set_x = set([sample2char(sample) for sample in expected_x])
    sample2loc = {sample2char(sample): idx for idx, sample in enumerate(expected_x)}

    count = 0
    for idx, iter_val in enumerate(it_array):
        # for every sample in this batch
        for j, sample in enumerate(iter_val['X']):
            txt_sample = sample2char(sample)
            assert txt_sample in set_x  # check sequence is valid
            set_x.discard(txt_sample)
            if (idx * batch_size + j) == sample2loc[txt_sample]:
                count += 1
    assert count < (len(expected_x) * .5)  # check shuffle happened
    assert len(set_x) == 0      # check every sequence appeared in iterator


@pytest.mark.parametrize("strides", [3, 8])
def test_rolling_window(input_seq, batch_size, seq_len, strides):
    # This test checks if the rolling window works
    # We check if the first two samples in each batch are strided by strides

    # Truncate input sequence such that last section that doesn't fit in a batch
    # is thrown away
    input_seq = input_seq[:seq_len * batch_size * (len(input_seq) // seq_len // batch_size)]
    data_array = {'X': input_seq,
                  'y': np.roll(input_seq, axis=0, shift=-1)}
    time_steps = seq_len
    it_array = SequentialArrayIterator(data_arrays=data_array, time_steps=time_steps,
                                       stride=strides, batch_size=batch_size, tgt_key='y',
                                       shuffle=False)
    for idx, iter_val in enumerate(it_array):
        # Start of the array needs to be time_steps * idx
        assert np.array_equal(iter_val['X'][0, strides:time_steps],
                              iter_val['X'][1, :time_steps - strides])
        assert np.array_equal(iter_val['y'][0, strides:time_steps],
                              iter_val['y'][1, :time_steps - strides])
