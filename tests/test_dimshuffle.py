import pytest
import numpy as np

import geon as be
from geon.util.derivative_check import check_derivative
from geon.util.utils import executor
from geon.util.utils import RandomTensorGenerator

rng = RandomTensorGenerator(0, np.float32)


def test_dimshuffle_fprop():
    """
    dimshuffle a 2d array and make sure fprop works
    """
    A = be.Axis(2)
    B = be.Axis(3)

    x = be.placeholder(axes=be.Axes([A, B]))

    # compute convolution with graph
    output = be.dimshuffle(x, axes=be.Axes([B, A]))

    assert output.axes == be.Axes([B, A])

    # randomly initialize
    x_value = rng.uniform(-1, 1, x.axes)

    result = executor(output, x)(x_value)

    np.testing.assert_allclose(result, x_value.T)


def test_dimshuffle_bprop():
    """
    dimshuffle a 2d array and make sure bprop works
    """
    A = be.Axis(2)
    B = be.Axis(3)

    x = be.placeholder(axes=be.Axes([A, B]))

    # randomly initialize
    x_value = rng.uniform(-1, 1, x.axes)

    check_derivative(
        be.dimshuffle(x, axes=be.Axes([B, A])),
        x, 0.001, x_value,
        atol=1e-3, rtol=1e-3
    )


def test_dimshuffle_error():
    A = be.Axis(2)
    B = be.Axis(3)
    C = be.Axis(4)

    x = be.placeholder(axes=be.Axes([A, B]))

    # fail on missing A
    with pytest.raises(ValueError):
        be.dimshuffle(x, axes=be.Axes([B, B]))

    # fail on extra Axis
    with pytest.raises(ValueError):
        be.dimshuffle(x, axes=be.Axes([A, B, C]))

    # fail on missing B, extra C
    with pytest.raises(ValueError):
        be.dimshuffle(x, axes=be.Axes([A, C]))

    # fail on reuse of B
    with pytest.raises(ValueError):
        be.dimshuffle(x, axes=be.Axes([A, B, B]))
