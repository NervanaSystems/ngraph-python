import numpy as np

import geon as be
from geon.util.derivative_check import check_derivative
from geon.util.utils import executor
from geon.util.utils import RandomTensorGenerator

rng = RandomTensorGenerator(0, np.float32)


def test_print_op_bprop():
    """
    Ensure bprop of PrintOp is correct (passes through exactly the delta)
    """

    A = be.Axis(10, name='A')

    x = be.placeholder(axes=be.Axes([A]))

    # randomly initialize
    x_value = rng.uniform(-1, 1, x.axes)

    check_derivative(
        be.PrintOp(x),
        x, 0.001, x_value,
        atol=1e-3, rtol=1e-3
    )


def test_print_op_fprop(capfd):
    """
    Ensure fprop of PrintOp makes no change to input, and also prints to
    stdout.
    """

    A = be.Axis(1, name='A')

    x = be.placeholder(axes=be.Axes([A]))

    # hardcode value so there are is no rounding to worry about in str
    # comparison in  final assert
    x_value = np.array([1])

    output = be.PrintOp(x, 'prefix')
    result = executor(output, x)(x_value)

    np.testing.assert_allclose(result, x_value)

    out, err = capfd.readouterr()
    assert str(x_value[0]) in out
    assert 'prefix' in out
