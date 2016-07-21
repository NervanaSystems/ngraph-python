import numpy as np
import random
import geon.backends.graph.funs as be
import geon.backends.graph.axis as ax
from geon.backends.graph.graph_test_utils import *

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
        }
    ]
    for test in tests:
        for axis, length in test['axes_lengths'].items():
            axis.length = length

        tensor1_axes = test['tensor1_axes']
        tensor1_np = be.NumPyTensor(
            np.array(test['tensor1'], dtype=np.float32), axes=tensor1_axes)
        tensor1 = be.Variable(axes=tensor1_axes, initial_value=tensor1_np)
        tensor2_axes = test['tensor2_axes']
        tensor2_np = be.NumPyTensor(
            np.array(test['tensor2'], dtype=np.float32), axes=tensor2_axes)
        tensor2 = be.Variable(axes=tensor2_axes, initial_value=tensor2_np)
        expected_output = np.array(test['expected_output'], dtype=np.float32)

        dot = be.dot(tensor1, tensor2)

        assert np.array_equal(evaluate(dot), expected_output)


if __name__ == '__main__':
    test_l2_norm()
    test_tensor_dot_tensor()
