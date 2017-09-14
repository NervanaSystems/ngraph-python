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


@pytest.mark.parametrize("shuffle", [True, False])
def test_nowindow(input_seq, batch_size, seq_len, shuffle):
    """
    This test checks shuffle and no-shuffle option, with non-overlapping windows
    We check only the first sample in each batch
    """
    data_array = {'X': input_seq,
                  'y': np.roll(input_seq, axis=0, shift=-1)}
    time_steps = seq_len
    it_array = SequentialArrayIterator(data_arrays=data_array, time_steps=time_steps,
                                       batch_size=batch_size, tgt_key='y', shuffle=shuffle)
    for idx, iter_val in enumerate(it_array):
        # Start of the array needs to be time_steps * idx * batch_size
        start_idx = time_steps * idx * batch_size
        idcs = np.arange(start_idx, start_idx + time_steps * batch_size) % input_seq.shape[0]
        reshape_dims = (batch_size, time_steps)
        if len(input_seq.shape) > 1:
            reshape_dims = (batch_size, time_steps, input_seq.shape[1])
        # expected_x will have contigous non-overlapping samples
        expected_x = data_array['X'][idcs].reshape(reshape_dims)
        expected_y = data_array['y'][idcs].reshape(reshape_dims)

        if not shuffle:
            # If we are not shuffling, we need to check consecutive samples are contiguous
            # They will also be non-overlapping
            assert np.array_equal(expected_x, iter_val['X'])
            assert np.array_equal(expected_y, iter_val['y'])
        else:
            # If we are shuffling, make sure the consecutive samples are not contiguous
            assert not np.array_equal(expected_x, iter_val['X'])
            assert not np.array_equal(expected_y, iter_val['y'])


@pytest.mark.parametrize("strides", [3, 8])
def test_rolling_window(input_seq, batch_size, seq_len, strides):
    # This test checks if the rolling window works
    # We check if the first two samples in each batch are strided by strides
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
