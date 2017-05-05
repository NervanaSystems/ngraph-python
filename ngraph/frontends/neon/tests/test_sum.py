import pytest
import numpy as np
import ngraph as ng


np.random.seed(0)


@pytest.fixture(params=[50, 75, 100])
def sequence_length(request):
    return request.param


@pytest.fixture(params=[16])
def batch_size(request):
    return request.param


@pytest.fixture(params=[2, 4])
def num_units(request):
    return request.param


@pytest.fixture(params=[0, 1])
def extra_axes(request):
    return request.param


def test_sum(transformer_factory, num_units, sequence_length, batch_size, extra_axes):
    """
    There is a non-deterministic error in ng.sum with the gpu transformer. The test below
    should show it.
    :param transformer_factory:
    :return:
    """

    # Mimic the output of a convolutional layer (if extra_axes > 0)
    # This doesn't seem to matter like I thought it did.
    shape = tuple([num_units] + [1] * extra_axes + [sequence_length, batch_size])
    np_inp = np.random.uniform(-1, 1, shape)

    # Use an identity weight matrix on top of it
    np_w = np.eye(shape[0])
    for _ in range(len(shape) - 3):
        np_w = np.expand_dims(np_w, 2)

    # Create ngraph versions
    inp = ng.constant(np_inp)
    reduction_axes = inp.axes[:-2]
    other_axes = inp.axes[-2:]
    new_axis = ng.make_axis(length=shape[0])
    w_axes = ng.make_axes(new_axis) | reduction_axes
    w = ng.constant(np_w, axes=w_axes)

    # Reshape to do similar dot in numpy
    inp_reshape = np.reshape(np_inp, (np.prod(reduction_axes.lengths),
                                      np.prod(other_axes.lengths)))
    w_reshape = np.reshape(np_w, (new_axis.length, inp_reshape.shape[0]))

    # Reduce dimensions with identity weight matrix
    np_x = np.dot(w_reshape, inp_reshape)
    x = ng.dot(w, inp)

    # Sum over all but the first axis
    output_axes = ng.make_axes(x.axes[0])
    y = ng.sum(x, out_axes=output_axes)
    np_y = np.sum(np_x, axis=1)

    t = transformer_factory()
    f = t.computation([y, x])
    y_val, x_val = f()

    assert np.allclose(x_val.ravel(),
                       np_x.ravel(),
                       atol=1e-5), "Max difference: {}".format(np.max(np.abs(x_val - np_x)))
    assert np.allclose(y_val,
                       np_y,
                       atol=1e-5), "Max difference: {}".format(np.max(np.abs(y_val - np_y)))
