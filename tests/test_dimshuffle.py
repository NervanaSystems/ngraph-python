import pytest
import numpy as np

import ngraph as ng
from ngraph.util.derivative_check import check_derivative
from ngraph.util.utils import executor
from ngraph.util.utils import RandomTensorGenerator

rng = RandomTensorGenerator(0, np.float32)


def test_dimshuffle_fprop():
    """
    dimshuffle a 2d array and make sure fprop works
    """
    A = ng.Axis(2)
    B = ng.Axis(3)

    x = ng.placeholder(axes=ng.Axes([A, B]))

    # compute convolution with graph
    output = ng.dimshuffle(x, axes=ng.Axes([B, A]))

    assert output.axes == ng.Axes([B, A])

    # randomly initialize
    x_value = rng.uniform(-1, 1, x.axes)

    result = executor(output, x)(x_value)

    np.testing.assert_allclose(result, x_value.T)


def test_dimshuffle_bprop():
    """
    dimshuffle a 2d array and make sure bprop works
    """
    A = ng.Axis(2)
    B = ng.Axis(3)

    x = ng.placeholder(axes=ng.Axes([A, B]))

    # randomly initialize
    x_value = rng.uniform(-1, 1, x.axes)

    check_derivative(
        ng.dimshuffle(x, axes=ng.Axes([B, A])),
        x, 0.001, x_value,
        atol=1e-3, rtol=1e-3
    )


@pytest.fixture
def A():
    return ng.Axis(2)


@pytest.fixture
def B():
    return ng.Axis(3)


@pytest.fixture
def C():
    return ng.Axis(4)


@pytest.fixture
def x(A, B, C):
    return ng.placeholder(axes=ng.Axes([A, B]))


def test_fail_on_missing(x, B):
    with pytest.raises(ValueError):
        ng.dimshuffle(x, axes=ng.Axes([B, B]))


def test_fail_on_extra_axis(x, A, B, C):
    with pytest.raises(ValueError):
        ng.dimshuffle(x, axes=ng.Axes([A, B, C]))


def test_fail_on_missing_and_extra_axis(x, A, C):
    with pytest.raises(ValueError):
        ng.dimshuffle(x, axes=ng.Axes([A, C]))


def test_fail_on_axis_reuse(x, A, B):
    with pytest.raises(ValueError):
        ng.dimshuffle(x, axes=ng.Axes([A, B, B]))
