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

import ngraph as ng
from ngraph.util.derivative_check import check_derivative
from ngraph.util.utils import executor
from ngraph.util.utils import RandomTensorGenerator

rng = RandomTensorGenerator(0, np.float32)


def test_print_op_bprop():
    """
    Ensure bprop of PrintOp is correct (passes through exactly the delta)
    """

    A = ng.make_axis(10, name='A')

    x = ng.placeholder(axes=ng.make_axes([A]))

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

    A = ng.make_axis(1, name='A')

    x = ng.placeholder(axes=ng.make_axes([A]))

    # hardcode value so there are is no rounding to worry about in str
    # comparison in  final assert
    x_value = np.array([1])

    output = ng.PrintOp(x, 'prefix')
    result = executor(output, x)(x_value)

    np.testing.assert_allclose(result, x_value)

    out, err = capfd.readouterr()
    assert str(x_value[0]) in out
    assert 'prefix' in out
