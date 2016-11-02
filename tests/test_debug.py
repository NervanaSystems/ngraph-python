import numpy as np

import ngraph as ng
from ngraph.util.derivative_check import check_derivative
from ngraph.util.utils import executor
from ngraph.util.utils import RandomTensorGenerator

rng = RandomTensorGenerator(0, np.float32)


def test_print_op_bprop():
    """
    Ensure bprop of PrintOp is correct (passes through exactly the delta)
    """

    A = ng.makeAxis(10, name='A')

    x = ng.placeholder(axes=ng.makeAxes([A]))

    # randomly initialize
    x_value = rng.uniform(-1, 1, x.axes)

    check_derivative(
        ng.PrintOp(x),
        x, 0.001, x_value,
        atol=1e-3, rtol=1e-3
    )


def test_print_op_fprop(capfd):
    """
    Ensure fprop of PrintOp makes no change to input, and also prints to
    stdout.
    """

    A = ng.makeAxis(1, name='A')

    x = ng.placeholder(axes=ng.makeAxes([A]))

    # hardcode value so there are is no rounding to worry about in str
    # comparison in  final assert
    x_value = np.array([1])

    output = ng.PrintOp(x, 'prefix')
    result = executor(output, x)(x_value)

    np.testing.assert_allclose(result, x_value)

    out, err = capfd.readouterr()
    assert str(x_value[0]) in out
    assert 'prefix' in out
