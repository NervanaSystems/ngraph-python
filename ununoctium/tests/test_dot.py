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
import random
import geon.backends.graph.funs as be
import geon.backends.graph.axis as ax
from geon.backends.graph.graph_test_utils import\
    in_bound_environment, raise_all_numpy_errors,\
    transform_numeric_derivative,\
    transform_derivative

'''
Test graphiti's implementation of the dot product.

'''


def evaluate(result):
    return be.NumPyTransformer(results=[result]).evaluate()[result]


def get_random_shape(max_num_axes, max_axis_length):
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
    arr = sigma * \
        np.random.randn(
            *get_random_shape(max_num_axes, max_axis_length)) + mean
    arr.dtype = dtype
    return arr


@in_bound_environment
def graphiti_l2_norm(np_array):
    axes = ()
    for i, l in enumerate(np_array.shape):
        axes += (be.AxisVar(name='axis%s' % i, length=l),)
    np_tensor = be.NumPyTensor(np_array, axes=axes)
    var = be.Variable(axes=axes, initial_value=np_tensor)
    return evaluate(be.sqrt(be.dot(var, var)))


@raise_all_numpy_errors
def test_l2_norm():
    tests_ = [
        [],
        [1],
        [1, 4, 5, 6],
        [[1, 3, 5], [4, 2, 5]]
    ]
    tests = map(lambda _: np.array(_, dtype=np.float32), tests_)
    for i, test in enumerate(tests):
        assert np.array_equal(np.linalg.norm(test), graphiti_l2_norm(test))


@raise_all_numpy_errors
@in_bound_environment
def test_tensor_dot_tensor():
    delta = rtol = atol = 1e-3

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
    ]

    print 'Testing the dot product implementation along with its autodiff.'
    for i, test in enumerate(tests):
        for axis, length in test['axes_lengths'].items():
            axis.length = length

        tensor1_axes = test['tensor1_axes']
        tensor1 = be.placeholder(axes=tensor1_axes)
        tensor1.value = np.array(
            test['tensor1'], dtype=np.float32
        )

        tensor2_axes = test['tensor2_axes']
        tensor2 = be.placeholder(axes=tensor2_axes)
        tensor2.value = np.array(
            test['tensor2'], dtype=np.float32
        )
        expected_output = np.array(test['expected_output'], dtype=np.float32)

        dot = be.dot(tensor1, tensor2)
        assert np.array_equal(evaluate(dot), expected_output)

        numeric_deriv_1 = transform_numeric_derivative(dot, tensor1, delta)
        sym_deriv_1 = transform_derivative(dot, tensor1)
        assert np.allclose(numeric_deriv_1, sym_deriv_1, rtol=rtol, atol=atol)

        numeric_deriv_2 = transform_numeric_derivative(dot, tensor2, delta)
        sym_deriv_2 = transform_derivative(dot, tensor2)
        assert np.allclose(numeric_deriv_2, sym_deriv_2, rtol=rtol, atol=atol)

        print 'Passed test %s.' % (i + 1)

if __name__ == '__main__':
    test_l2_norm()
    test_tensor_dot_tensor()
