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
import random

import ngraph as ng
import numpy as np
from builtins import range
from ngraph.util.utils import ExecutorFactory, executor
from ngraph.util.utils import raise_all_numpy_errors

"""
Test ngraph's implementation of the dot product.
"""


def get_random_shape(max_num_axes, max_axis_length):
    """
    TODO.

    Arguments:
      max_num_axes: TODO
      max_axis_length: TODO

    Returns:
      TODO
    """
    assert max_num_axes >= 2
    num_axes = 0

    while num_axes < 2:
        num_axes = random.randint(0, max_num_axes)
        shape = ()
        for i in range(num_axes):
            shape += (random.randint(0, max_axis_length),)
    return shape


def get_random_np_array(
        max_num_axes,
        max_axis_length,
        mean=0,
        sigma=1,
        dtype=np.float32):
    """
    TODO.

    Arguments:
      max_num_axes: TODO
      max_axis_length: TODO
      mean: TODO
      sigma: TODO
      dtype: TODO

    Returns:
      TODO
    """
    shape = get_random_shape(max_num_axes, max_axis_length)
    arr = sigma * np.random.randn(*shape) + mean
    arr.dtype = dtype
    return arr


def ngraph_l2_norm(np_array):
    """
    TODO.

    Arguments:
      np_array: TODO

    Returns:
      TODO
    """
    axes = ()
    for i, l in enumerate(np_array.shape):
        axes += (ng.make_axis(name='axis%s' % i, length=l),)

    np_tensor = ng.constant(np_array, axes)
    var = ng.variable(axes, initial_value=np_tensor)
    return executor(ng.sqrt(ng.dot(var, var)))()


@raise_all_numpy_errors
def test_dot_sum_backprop(transformer_factory):
    delta = 1e-3
    rtol = atol = 1e-2

    C = ng.make_axis(name='C')
    N = ng.make_axis(name='N')

    C.length = 2
    N.length = 3
    N.batch = True
    x_axes, y_axes = ng.make_axes((C, N)), ng.make_axes((C,))
    x_np = np.random.random(x_axes.lengths).astype('float32')
    y_np = np.random.random(y_axes.lengths).astype('float32')
    expected_output = np.sum(x_np.T.dot(y_np))

    x = ng.placeholder(x_axes)
    y = ng.placeholder(y_axes)
    d = ng.dot(x, y)
    s = ng.sum(d, out_axes=())

    ex = ExecutorFactory()
    evaluated_fun = ex.executor(s, x, y)
    numeric_deriv_fun1 = ex.numeric_derivative(s, x, delta, y)
    numeric_deriv_fun2 = ex.numeric_derivative(s, y, delta, x)
    sym_deriv_fun1 = ex.derivative(s, x, y)
    sym_deriv_fun2 = ex.derivative(s, y, x)

    # assert outputs are equal
    evaluated = evaluated_fun(x_np, y_np)
    np.testing.assert_allclose(evaluated, expected_output, rtol=rtol, atol=atol)

    # assert derivative wrt to both tensors is the same when computed
    # symbolicly by ngraph and numerically
    numeric_deriv1 = numeric_deriv_fun1(x_np, y_np)
    sym_deriv1 = sym_deriv_fun1(x_np, y_np)
    np.testing.assert_allclose(numeric_deriv1, sym_deriv1, rtol=rtol, atol=atol)

    numeric_deriv2 = numeric_deriv_fun2(y_np, x_np)
    sym_deriv2 = sym_deriv_fun2(y_np, x_np)
    np.testing.assert_allclose(numeric_deriv2, sym_deriv2, rtol=rtol, atol=atol)


@raise_all_numpy_errors
def test_tensor_dot_tensor(transformer_factory):
    """TODO."""
    C = ng.make_axis(name='C')
    D = ng.make_axis(name='D')
    H = ng.make_axis(name='H')
    N = ng.make_axis(name='N')

    tests = [
        {
            'tensor1': [[1, 2], [4, 5], [3, 4]],
            'tensor1_axes': (C, D),
            'tensor2': [2, 5],
            'tensor2_axes': (D,),
            'expected_output': [12, 33, 26],
            'axes_lengths': {C: 3, D: 2}
        },
        {
            'tensor1': [[1, 4, 3], [2, 5, 4]],
            'tensor1_axes': (D, C),
            'tensor2': [2, 5],
            'tensor2_axes': (D,),
            'expected_output': [12, 33, 26],
            'axes_lengths': {C: 3, D: 2}
        },
        {
            'tensor1': [[[1, 4], [2, 5]], [[7, 12], [13, 2]]],
            'tensor1_axes': (N, D, C),
            'tensor2': [[[3, 6], [7, 2]], [[9, 8], [10, 4]]],
            'tensor2_axes': (H, D, C),
            'expected_output': [[51, 81], [188, 297]],
            'axes_lengths': {N: 2, D: 2, C: 2, H: 2}
        },
        {
            'tensor1': [1, 2],
            'tensor1_axes': (C,),
            'tensor2': [7, 11, 13],
            'tensor2_axes': (D,),
            'expected_output': [[7, 11, 13], [14, 22, 26]],
            'axes_lengths': {C: 2, D: 3}
        },
        {
            'tensor1': [[1, 4], [6, 2]],
            'tensor1_axes': (C, D),
            'expected_output': 57,
            'axes_lengths': {C: 2, D: 2}
        }
    ]

    for test in tests:
        # set up axis
        for axis, length in test['axes_lengths'].items():
            axis.length = length

        # set up tensors
        tensor1 = ng.placeholder(test['tensor1_axes'])
        value1 = np.array(test['tensor1'], dtype=np.float32)

        if 'tensor2' in test:
            tensor2 = ng.placeholder(test['tensor2_axes'])
            value2 = np.array(
                test['tensor2'], dtype=np.float32
            )
        else:
            tensor2 = tensor1
            value2 = value1

        # compute outputs
        expected_output = np.array(test['expected_output'], dtype=np.float32)

        ex = ExecutorFactory()
        dot = ng.dot(tensor1, tensor2)
        evaluated_fun = ex.executor(dot, tensor1, tensor2)

        numeric_deriv_fun1 = ex.numeric_derivative(dot, tensor1, 1e-3, tensor2)
        numeric_deriv_fun2 = ex.numeric_derivative(dot, tensor2, 1e-3, tensor1)
        sym_deriv_fun1 = ex.derivative(dot, tensor1, tensor2)
        sym_deriv_fun2 = ex.derivative(dot, tensor2, tensor1)

        # assert outputs are equal
        evaluated = evaluated_fun(value1, value2)
        np.testing.assert_equal(evaluated, expected_output)

        # assert derivative wrt to both tensors is the same when computed
        # symbolicly by ngraph and numerically
        numeric_deriv1 = numeric_deriv_fun1(value1, value2)
        sym_deriv1 = sym_deriv_fun1(value1, value2)
        np.testing.assert_allclose(numeric_deriv1, sym_deriv1, rtol=1e-2, atol=1e-2)

        numeric_deriv2 = numeric_deriv_fun2(value2, value1)
        sym_deriv2 = sym_deriv_fun2(value2, value1)
        np.testing.assert_allclose(numeric_deriv2, sym_deriv2, rtol=1e-2, atol=1e-2)
