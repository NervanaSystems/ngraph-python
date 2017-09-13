import pytest
import numpy as np
from ngraph.frontends.neon import SequentialArrayIterator


@pytest.fixture(scope='module',
                params=[
                    np.arange(1000),
                    np.random.rand(2001,)
                ],
                ids=[
                    'short_seq',
                    'long_seq'
                ])
def input_seq(request):
    return request.param


@pytest.fixture(scope='module',
                params=[
                    4,
                    32
                ])
def batches(request):
    return request.param


@pytest.fixture(scope='module',
                params=[
                    8,
                    16
                ])
def seq_len(request):
    return request.param


@pytest.fixture(scope='module',
                params=[
                    3,
                    8
                ])
def strides(request):
    return request.param


def test_noshuffle(input_seq, batches, seq_len):
    """
    This test checks no shuffle option, with non-overlapping windows
    We check only the first sample in each batch
    """
    data_array = {'X': input_seq,
                  'y': np.roll(input_seq, -1)}
    time_steps = seq_len
    batch_size = batches
    it_array = SequentialArrayIterator(data_arrays=data_array, time_steps=time_steps,
                                       batch_size=batch_size, tgt_key='y', shuffle=False)
    idx = 0
    expected = {'X': np.zeros((time_steps,), dtype=data_array['X'].dtype),
                'y': np.zeros((time_steps,), dtype=data_array['y'].dtype)}
    for iter_val in it_array:
        # Start of the array needs to be time_steps * idx * batch_size
        start_idx = time_steps * idx * batch_size
        idcs = np.arange(start_idx, start_idx + time_steps) % input_seq.shape[0]
        expected['X'] = data_array['X'][idcs]
        expected['y'] = data_array['y'][idcs]
        assert np.array_equal(expected['X'], iter_val['X'][0, :])
        assert np.array_equal(expected['y'], iter_val['y'][0, :])
        idx += 1


def test_shuffle(input_seq, batches, seq_len):
    # This test checks the shuffle option, with non-overlapping windows
    # We check only the first sample in each batch
    data_array = {'X': input_seq,
                  'y': np.roll(input_seq, -1)}
    time_steps = seq_len
    batch_size = batches
    it_array = SequentialArrayIterator(data_arrays=data_array, time_steps=time_steps,
                                       batch_size=batch_size, tgt_key='y', shuffle=True)
    idx = 0
    expected = {'X': np.zeros((time_steps,), dtype=data_array['X'].dtype),
                'y': np.zeros((time_steps,), dtype=data_array['y'].dtype)}
    for iter_val in it_array:
        # Start of the array needs to be time_steps * idx
        start_idx = time_steps * idx
        idcs = np.arange(start_idx, start_idx + time_steps) % input_seq.shape[0]
        expected['X'] = data_array['X'][idcs]
        expected['y'] = data_array['y'][idcs]
        assert np.array_equal(expected['X'], iter_val['X'][0, :])
        assert np.array_equal(expected['y'], iter_val['y'][0, :])
        idx += 1


def test_rolling_window(input_seq, batches, seq_len, strides):
    # This test checks if the rolling window works
    # We check if the first two samples in each batch are strided by strides
    data_array = {'X': input_seq,
                  'y': np.roll(input_seq, -1)}
    time_steps = strides
    batch_size = batches
    it_array = SequentialArrayIterator(data_arrays=data_array, time_steps=time_steps,
                                       batch_size=batch_size, tgt_key='y', shuffle=False)
    for iter_val in it_array:
        # Start of the array needs to be time_steps * idx
        assert np.array_equal(iter_val['X'][0, strides:time_steps],
                              iter_val['X'][1, :time_steps - strides])
        assert np.array_equal(iter_val['y'][0, strides:time_steps],
                              iter_val['y'][1, :time_steps - strides])
