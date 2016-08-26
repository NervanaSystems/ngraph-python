import numpy as np

from geon.util.utils import ExecutorFactory


def check_derivative(f, x, delta, x_value, parameters=[], parameter_values=[], **kwargs):
    """
    Check that the numeric and symbol derivatives of f with respect to x are
    the same when x has value x_value.

    Arguments:
        f: function to take the derivative of
        x: variable to take the derivative with respect to
        delta: distance to perturn x in numeric derivative
        x_value: the value of x we are going to compute the derivate of f at
        parameters: extra parameters to f
        parameter_values: value of extra parameters to f
        kwargs: passed to assert_allclose.  Useful for atol/rtol.
    """

    ex = ExecutorFactory()

    dfdx_numeric = ex.numeric_derivative(f, x, delta, *parameters)
    dfdx_symbolic = ex.derivative(f, x, *parameters)

    np.testing.assert_allclose(
        dfdx_numeric(x_value, *parameter_values),
        dfdx_symbolic(x_value, *parameter_values),
        **kwargs
    )
