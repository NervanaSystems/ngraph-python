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
import pytest
import numpy as np

from ngraph.util.utils import ExecutorFactory
import ngraph as ng
import ngraph.frontends.base.axis as ax
from ngraph.op_graph import arrayaxes


delta = 1e-3
rtol = atol = 1e-2


def test_expand_dims():
    """TODO."""
    max_new_axis_length = 4

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
                ex = ExecutorFactory()
                for axis, length in zip(tensor_axes, tensor_axes_lengths):
                    axis.length = length

                new_axis = test['new_axis']
                new_axis.length = new_axis_length

                tensor_np = np.array(
                    test['tensor'], dtype=np.float32
                )
                tensor = ng.placeholder(axes=ng.Axes(tensor_axes))

                expanded = ng.ExpandDims(tensor, new_axis, dim)
                expander_fun = ex.executor(expanded, tensor)

                num_deriv_fun = ex.numeric_derivative(expanded, tensor, delta)
                sym_deriv_fun = ex.derivative(expanded, tensor)

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

                expanded_result = expander_fun(tensor_np)
                assert np.array_equal(expanded_np, expanded_result)

                # Test backpropagation
                numeric_deriv = num_deriv_fun(tensor_np)
                sym_deriv = sym_deriv_fun(tensor_np)
                assert np.allclose(
                    numeric_deriv, sym_deriv, rtol=rtol, atol=atol
                )


def test_slice():
    """TODO."""
    tests = [
        {
            'tensor': [[1, 3], [2, 5]],
            'tensor_axes': (ax.C, ax.D),
            'slice': [0, 1],
            'sliced_axes': (),
            'axes_lengths': {ax.C: 2, ax.D: 2},
            'expected': 3
        },
        {
            'tensor': [[1, 3], [2, 5]],
            'tensor_axes': (ax.C, ax.D),
            'slice': [slice(None), 0],
            'sliced_axes': (ax.C,),
            'axes_lengths': {ax.C: 2, ax.D: 2},
            'expected': [1, 2]
        },
        {
            'tensor': [[1, 3], [2, 5]],
            'tensor_axes': (ax.C, ax.D),
            'slice': [1, slice(None)],
            'sliced_axes': (ax.D,),
            'axes_lengths': {ax.C: 2, ax.D: 2},
            'expected': [2, 5]
        },
        {
            'tensor': [[1, 4, 5], [2, 5, 6]],
            'tensor_axes': (ax.C, ax.D),
            'slice': [1, slice(1, 3)],
            'sliced_axes': None,
            'axes_lengths': {ax.C: 2, ax.D: 3},
            'expected': [5, 6]
        },
        {
            'tensor': [[1, 4, 5], [2, 5, 6]],
            'tensor_axes': (ax.C, ax.D),
            'slice': [1, slice(None, None, -1)],
            'sliced_axes': None,
            'axes_lengths': {ax.C: 2, ax.D: 3},
            'expected': [6, 5, 2]
        },
        {
            'tensor': [[1, 4, 5], [2, 5, 6]],
            'tensor_axes': (ax.C, ax.D),
            'slice': [slice(None, None, -1), slice(None, None, -1)],
            'sliced_axes': None,
            'axes_lengths': {ax.C: 2, ax.D: 3},
            'expected': [[6, 5, 2], [5, 4, 1]]
        }
    ]

    for test in tests:
        ex = ExecutorFactory()
        for axis, length in test['axes_lengths'].items():
            axis.length = length
        tensor_axes = test['tensor_axes']

        tensor_np = np.array(
            test['tensor'], dtype='float32'
        )
        tensor = ng.placeholder(axes=ng.Axes(tensor_axes))
        expected = np.array(test['expected'], dtype='float32')

        s = test['slice']
        s_axes = test['sliced_axes']

        sliced = ng.Slice(tensor, s, s_axes)
        sliced_val_fun = ex.executor(sliced, tensor)

        num_deriv_fun = ex.numeric_derivative(sliced, tensor, delta)
        # Test backpropagation
        sym_deriv_fun = ex.derivative(sliced, tensor)

        sliced_val = sliced_val_fun(tensor_np)
        assert np.array_equal(sliced_val, expected)

        numeric_deriv = num_deriv_fun(tensor_np)
        sym_deriv = sym_deriv_fun(tensor_np)

        assert np.allclose(
            numeric_deriv, sym_deriv, rtol=rtol, atol=atol
        )


def test_padding():
    """TODO."""
    tests = [
        {
            'tensor': [[1, 3], [2, 5]],
            'tensor_axes': (ax.C, ax.D),
            'padding': [(0, 1), (1, 0)],
            'padded_axes': (ax.M, ax.N),
            'axes_lengths': {ax.C: 2, ax.D: 2, ax.M: 3, ax.N: 3}
        },
        {
            'tensor': [[1, 4, 5], [1, 4, 6]],
            'tensor_axes': (ax.C, ax.D),
            'padding': [(0, 1), 1],
            'padded_axes': None,
            'axes_lengths': {ax.C: 2, ax.D: 3}
        }
    ]

    for test in tests:
        ex = ExecutorFactory()
        for axis, length in test['axes_lengths'].items():
            axis.length = length
        tensor_axes = test['tensor_axes']
        tensor_np = np.array(
            test['tensor'], dtype='float32'
        )
        tensor = ng.placeholder(axes=ng.Axes(tensor_axes))
        padding = test['padding']
        padded_axes = test['padded_axes']
        padded = ng.pad(tensor, padding, padded_axes)
        computed_val_fun = ex.executor(padded, tensor)

        # Test backpropagation
        numeric_deriv_fun = ex.numeric_derivative(padded, tensor, delta)
        sym_deriv_fun = ex.derivative(padded, tensor)

        def to_tuple(p):
            """
            TODO.

            Arguments:
              p: TODO

            Returns:

            """
            return (p, p) if isinstance(p, int) else p
        np_padding = tuple(to_tuple(p) for p in padding)
        expected_val = np.pad(tensor_np, np_padding, mode='constant')

        computed_val = computed_val_fun(tensor_np)
        assert np.array_equal(expected_val, computed_val)

        numeric_deriv = numeric_deriv_fun(tensor_np)
        sym_deriv = sym_deriv_fun(tensor_np)

        assert np.allclose(
            numeric_deriv, sym_deriv, rtol=rtol, atol=atol
        )


def test_axes_cast():
    ex = ExecutorFactory()

    ax.C.length = 2
    ax.D.length = 3

    x = ng.placeholder(axes=(ax.C, ax.D))

    x_slice = x[1, :]
    # Cast back to known axes
    x_cast = x_slice.with_axes(ax.D)

    # Verfiy that the tensor broadcasts along ax.D
    y = x + x_cast
    y_fun = ex.executor(y, x)
    num_deriv_fun = ex.numeric_derivative(y, x, delta)
    sym_deriv_fun = ex.derivative(y, x)

    x_np = np.array([[10, 20, 30], [1, 2, 3]], dtype='float32')
    assert np.allclose(
        y_fun(x_np),
        np.array([[11, 22, 33], [2, 4, 6]], dtype='float32')
    )

    assert np.allclose(
        num_deriv_fun(x_np),
        sym_deriv_fun(x_np),
        rtol=rtol,
        atol=atol
    )


def test_slice_tensor_description():
    C = arrayaxes.Axis(2)

    td = arrayaxes.TensorDescription(arrayaxes.Axes(C))
    with pytest.raises(ValueError):
        td.slice(
            [slice(None)],
            arrayaxes.Axes([arrayaxes.Axis(1), arrayaxes.Axis(1)]),
        )


def test_tensor_description_init():
    with pytest.raises(ValueError):
        # TensorDescription axes require lengths
        arrayaxes.TensorDescription(arrayaxes.Axes(arrayaxes.Axis()))
