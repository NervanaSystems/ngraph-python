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
from builtins import range

import numpy as np

import ngraph as ng
from ngraph.testing import ExecutorFactory, executor
from ngraph.testing import raise_all_numpy_errors
import pytest

pytestmark = pytest.mark.transformer_dependent("module")


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
        axes |= (ng.make_axis(length=l).named('axis%s' % i),)

    np_tensor = ng.constant(np_array, axes)
    var = ng.variable(axes, initial_value=np_tensor)
    with executor(ng.sqrt(ng.squared_L2(var))) as ex:
        return ex()


@raise_all_numpy_errors
def test_dot_sum_backprop(transformer_factory):
    delta = 1e-3
    rtol = atol = 1e-2

    C = ng.make_axis(length=2).named('C')
    N = ng.make_axis(length=3, name='N')

    x_axes = ng.make_axes((C, N))
    y_axes = ng.make_axes((C,))
    x_np = np.random.random(x_axes.lengths).astype('float32')
    y_np = np.random.random(y_axes.lengths).astype('float32')

    x_np[...] = [[1.0, 0.0, 1.0], [2.0, 0.0, 3.0]]
    y_np[...] = [-1.0, 1.0]

    x = ng.placeholder(x_axes)
    y = ng.placeholder(y_axes)
    d = ng.dot(x, y)
    s = ng.sum(d, out_axes=())

    with ExecutorFactory() as ex:
        s_fun = ex.executor(s, x, y)
        d_fun = ex.executor(d, x, y)

        dd_dx_fun_num = ex.numeric_derivative(d, x, delta, y)
        dd_dx_fun_sym = ex.derivative(d, x, y)

        dd_dy_fun_num = ex.numeric_derivative(d, y, delta, x)
        dd_dy_fun_sym = ex.derivative(d, y, x)

        ds_dx_fun_num = ex.numeric_derivative(s, x, delta, y)
        ds_dx_fun_sym = ex.derivative(s, x, y)

        ds_dy_fun_num = ex.numeric_derivative(s, y, delta, x)
        ds_dy_fun_sym = ex.derivative(s, y, x)

        # assert outputs are equal
        d_np = x_np.T.dot(y_np)
        d_val = d_fun(x_np, y_np)
        ng.testing.assert_allclose(d_np, d_val, rtol=rtol, atol=atol)

        dd_dx_val_num = dd_dx_fun_num(x_np, y_np)
        dd_dx_val_sym = dd_dx_fun_sym(x_np, y_np)
        ng.testing.assert_allclose(dd_dx_val_num, dd_dx_val_sym, rtol=rtol, atol=atol)

        dd_dy_val_num = dd_dy_fun_num(y_np, x_np)
        dd_dy_val_sym = dd_dy_fun_sym(y_np, x_np)
        ng.testing.assert_allclose(dd_dy_val_num, dd_dy_val_sym, rtol=rtol, atol=atol)

        s_np = np.sum(d_np)
        s_val = s_fun(x_np, y_np)
        ng.testing.assert_allclose(s_val, s_np, rtol=rtol, atol=atol)

        # assert derivative wrt to both tensors is the same when computed
        # symbolically by ngraph and numerically
        ds_dx_val_num = ds_dx_fun_num(x_np, y_np)
        ds_dx_val_sym = ds_dx_fun_sym(x_np, y_np)
        ng.testing.assert_allclose(ds_dx_val_num, ds_dx_val_sym, rtol=rtol, atol=atol)

        ds_dy_val_num = ds_dy_fun_num(y_np, x_np)
        ds_dy_val_sym = ds_dy_fun_sym(y_np, x_np)
        ng.testing.assert_allclose(ds_dy_val_num, ds_dy_val_sym, rtol=rtol, atol=atol)


@pytest.mark.flex_disabled
@raise_all_numpy_errors
def test_tensor_dot_tensor(transformer_factory):
    """TODO."""
    C = ng.make_axis().named('C')
    D = ng.make_axis().named('D')
    H = ng.make_axis().named('H')
    N = ng.make_axis().named('N')

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
            'tensor2': [[1, 4], [6, 2]],
            'tensor2_axes': (C, D),
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

        tensor2 = ng.placeholder(test['tensor2_axes'])
        value2 = np.array(
            test['tensor2'], dtype=np.float32
        )

        # compute outputs
        expected_output = np.array(test['expected_output'], dtype=np.float32)

        with ExecutorFactory() as ex:
            dot = ng.dot(tensor1, tensor2)
            evaluated_fun = ex.executor(dot, tensor1, tensor2)

            deriv1_fun_num = ex.numeric_derivative(dot, tensor1, 1e-3, tensor2)
            deriv1_fun_sym = ex.derivative(dot, tensor1, tensor2)

            deriv2_fun_num = ex.numeric_derivative(dot, tensor2, 1e-3, tensor1)
            deriv2_fun_sym = ex.derivative(dot, tensor2, tensor1)

            # assert outputs are equal
            evaluated = evaluated_fun(value1, value2)
            np.testing.assert_equal(evaluated, expected_output)

            # assert derivative wrt to both tensors is the same when computed
            # symbolically by ngraph and numerically
            deriv1_val_num = deriv1_fun_num(value1, value2)
            deriv1_val_sym = deriv1_fun_sym(value1, value2)
            ng.testing.assert_allclose(deriv1_val_num, deriv1_val_sym, rtol=1e-2, atol=1e-2)

            deriv2_val_num = deriv2_fun_num(value2, value1)
            deriv2_val_sym = deriv2_fun_sym(value2, value1)
            ng.testing.assert_allclose(deriv2_val_num, deriv2_val_sym, rtol=1e-2, atol=1e-2)


def test_flat_tensor_dot_tensor(transformer_factory):
    """
    Ensure that a flattened argument axis is not unflattend in the result.

    Args:
        transformer_factory:

    Returns:
    """
    H = ng.make_axis(2)
    W = ng.make_axis(7)
    C = ng.make_axis(3)
    K = ng.make_axis(11)

    axes_a = ng.make_axes([H, W, C])
    a = ng.constant(np.ones(axes_a.lengths), axes=axes_a)
    flat_a = ng.flatten_at(a, 2)

    axes_b = ng.make_axes([C, K])
    b = ng.constant(np.ones(axes_b.lengths), axes=axes_b)

    result = ng.dot(b, flat_a)

    with ExecutorFactory() as factory:
        result_fun = factory.executor(result)
        result_val = result_fun()

    result_correct = np.ones_like(result_val) * C.length
    ng.testing.assert_allclose(result_val, result_correct)


def test_squared_L2(transformer_factory):
    H = ng.make_axis(2)
    W = ng.make_axis(3)
    N = ng.make_axis(5, name='N')

    axes = ng.make_axes([H, W, N])
    a = ng.constant(np.ones(axes.lengths), axes=axes)

    with ExecutorFactory() as factory:
        l2_samples_fun = factory.executor(ng.squared_L2(a))
        l2_samples_val = np.ones([N.length]) * H.length * W.length
        l2_all_fun = factory.executor(ng.squared_L2(a, out_axes=[]))
        l2_all_val = np.ones([]) * W.length * H.length * N.length
        l2_W_fun = factory.executor(ng.squared_L2(a, reduction_axes=[H, N]))
        l2_W_val = np.ones([W.length]) * H.length * N.length
        l2_samples_result = l2_samples_fun()
        l2_all_result = l2_all_fun()
        l2_W_result = l2_W_fun()
        ng.testing.assert_allclose(l2_samples_val, l2_samples_result)
        ng.testing.assert_allclose(l2_all_val, l2_all_result)
        ng.testing.assert_allclose(l2_W_val, l2_W_result)
