from geon.backends.graph.graph_test_utils import execute, be, Axes,\
    in_bound_environment, transform_numeric_derivative, transform_derivative
import geon.backends.graph.axis as ax
import numpy as np


@in_bound_environment
def test_expand_dims():
    max_new_axis_length = 4
    delta = 1e-3
    rtol = atol = 1e-2

    tests = [
        {
            'tensor': [[2, 5], [13, 5]],
            'tensor_axes': (ax.N, ax.D),
            'tensor_axes_lengths': (2, 2),
            'new_axis': ax.C,
        },
        {
            'tensor': 2,
            'tensor_axes': (),
            'tensor_axes_lengths': (),
            'new_axis': ax.D
        }
    ]

    for test in tests:
        for new_axis_length in range(1, max_new_axis_length + 1):
            tensor_axes = test['tensor_axes']
            tensor_axes_lengths = test['tensor_axes_lengths']

            for dim in range(len(tensor_axes) + 1):
                for axis, length in zip(tensor_axes, tensor_axes_lengths):
                    axis.length = length

                new_axis = test['new_axis']
                new_axis.length = new_axis_length

                tensor_np = np.array(
                    test['tensor'], dtype=np.float32
                )
                tensor = be.placeholder(axes=Axes(*tensor_axes))
                tensor.value = tensor_np

                expanded = be.ExpandDims(tensor, new_axis, dim)
                expanded_result, = execute([expanded])

                expanded_shape = tensor_np.shape[:dim] \
                    + (new_axis.length,) + tensor_np.shape[dim:]
                expanded_strides = tensor_np.strides[:dim] \
                    + (0,) + tensor_np.strides[dim:]
                expanded_np = np.ndarray(
                    buffer=tensor_np,
                    shape=expanded_shape,
                    strides=expanded_strides,
                    dtype=tensor_np.dtype
                )

                assert np.array_equal(expanded_np, expanded_result)

                # Test backpropagation
                numeric_deriv = transform_numeric_derivative(
                    expanded, tensor, delta
                )
                sym_deriv = transform_derivative(expanded, tensor)
                assert np.allclose(
                    numeric_deriv, sym_deriv, rtol=rtol, atol=atol
                )

if __name__ == '__main__':
    test_expand_dims()
