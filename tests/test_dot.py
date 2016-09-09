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
from builtins import range
import numpy as np
import random
import ngraph as ng
import ngraph.frontends.base.axis as ax
from ngraph.util.utils import raise_all_numpy_errors
from ngraph.util.utils import ExecutorFactory, executor

"""
Test graphiti's implementation of the dot product.
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


def graphiti_l2_norm(np_array):
    """
    TODO.

    Arguments:
      np_array: TODO

    Returns:
      TODO
    """
    axes = ()
    for i, l in enumerate(np_array.shape):
        axes +=  ng.Axis(name='axis%s' % i, length=l),)
    np_tensor = ng.Constant(np_array, axes=axes)
    var = ng.Variable(axes=axes, initial_value=np_tensor)
    return executor ng.sqrt ng.dot(var, var)))()


@raise_all_numpy_errors
def test_l2_norm():
    """TODO."""
    tests_ = [
        [],
        [1],
        [1, 4, 5, 6],
        [[1, 3, 5], [4, 2, 5]]
    ]
    tests = [np.array(_, dtype=np.float32) for _ in tests_]
    for test in tests:
        assert np.array_equal(np.linalg.norm(test), graphiti_l2_norm(test))


@raise_all_numpy_errors
def test_dot_sum_backprop():
    delta = 1e-3
    rtol = atol = 1e-2

    ax.C.length = 2
    ax.N.length = 3
    ax.N.batch = True
    x_axes, y_axes = ng.Axes((ax.C, ax.N)), ng.Axes((ax.C,))
    x_np = np.random.random(x_axes.lengths).astype('float32')
    y_np = np.random.random(y_axes.lengths).astype('float32')
    expected_output = np.sum(x_np.T.dot(y_np))

    x = ng.placeholder(axes=x_axes)
    y = ng.placeholder(axes=y_axes)
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
    np.testing.assert_equal(evaluated, expected_output)

    # assert derivative wrt to both tensors is the same when computed
    # symbolicly by graphiti and numerically
    numeric_deriv1 = numeric_deriv_fun1(x_np, y_np)
    sym_deriv1 = sym_deriv_fun1(x_np, y_np)
    np.testing.assert_allclose(numeric_deriv1, sym_deriv1, rtol=rtol, atol=atol)

    numeric_deriv2 = numeric_deriv_fun2(y_np, x_np)
    sym_deriv2 = sym_deriv_fun2(y_np, x_np)
    np.testing.assert_allclose(numeric_deriv2, sym_deriv2, rtol=rtol, atol=atol)


@raise_all_numpy_errors
def test_tensor_dot_tensor():
    """TODO."""
    delta = 1e-3
    rtol = atol = 1e-2

    tests = [
        {
            'tensor1': [[1, 2], [4, 5], [3, 4]],
            'tensor1_axes': (ax.C, ax.D),
            'tensor2': [2, 5],
            'tensor2_axes': (ax.D,),
            'expected_output': [12, 33, 26],
            'axes_lengths': {ax.C: 3, ax.D: 2}
        },
        {
            'tensor1': [[1, 4, 3], [2, 5, 4]],
            'tensor1_axes': (ax.D, ax.C),
            'tensor2': [2, 5],
            'tensor2_axes': (ax.D,),
            'expected_output': [12, 33, 26],
            'axes_lengths': {ax.C: 3, ax.D: 2}
        },
        {
            # Resembles hidden state to hidden state transformation
            'tensor1': [[1, 4], [2, 5]],
            'tensor1_axes': (ax.H, ax.H),
            'tensor2': [2, 5],
            'tensor2_axes': (ax.H,),
            'expected_output': [12, 33],
            'axes_lengths': {ax.H: 2}
        },
        {
            'tensor1': [[[1, 4], [2, 5]], [[7, 12], [13, 2]]],
            'tensor1_axes': (ax.N, ax.D, ax.C),
            'tensor2': [[[3, 6], [7, 2]], [[9, 8], [10, 4]]],
            'tensor2_axes': (ax.H, ax.D, ax.C),
            'expected_output': [[51, 81], [188, 297]],
            'axes_lengths': {ax.N: 2, ax.D: 2, ax.C: 2, ax.H: 2}
        },
        {
            'tensor1': [1, 2],
            'tensor1_axes': (ax.C,),
            'tensor2': [7, 11, 13],
            'tensor2_axes': (ax.D,),
            'expected_output': [[7, 11, 13], [14, 22, 26]],
            'axes_lengths': {ax.C: 2, ax.D: 3}
        },
        {
            'tensor1': [[1, 4], [6, 2]],
            'tensor1_axes': (ax.C, ax.D),
            'expected_output': 57,
            'axes_lengths': {ax.C: 2, ax.D: 2}
        }
    ]

    for test in tests:
        # set up axis
        for axis, length in test['axes_lengths'].items():
            axis.length = length

        # set up tensors
        tensor1 = ng.placeholder(axes=test['tensor1_axes'])
        value1 = np.array(
            test['tensor1'], dtype=np.float32
        )

        if 'tensor2' in test:
            tensor2 = ng.placeholder(axes=test['tensor2_axes'])
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

        numeric_deriv_fun1 = ex.numeric_derivative(dot, tensor1, delta, tensor2)
        numeric_deriv_fun2 = ex.numeric_derivative(dot, tensor2, delta, tensor1)
        sym_deriv_fun1 = ex.derivative(dot, tensor1, tensor2)
        sym_deriv_fun2 = ex.derivative(dot, tensor2, tensor1)

        # assert outputs are equal
        evaluated = evaluated_fun(value1, value2)
        np.testing.assert_equal(evaluated, expected_output)

        # assert derivative wrt to both tensors is the same when computed
        # symbolicly by graphiti and numerically
        numeric_deriv1 = numeric_deriv_fun1(value1, value2)
        sym_deriv1 = sym_deriv_fun1(value1, value2)
        np.testing.assert_allclose(numeric_deriv1, sym_deriv1, rtol=rtol, atol=atol)

        numeric_deriv2 = numeric_deriv_fun2(value2, value1)
        sym_deriv2 = sym_deriv_fun2(value2, value1)
        np.testing.assert_allclose(numeric_deriv2, sym_deriv2, rtol=rtol, atol=atol)
