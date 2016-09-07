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
