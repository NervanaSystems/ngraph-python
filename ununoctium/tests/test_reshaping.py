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

from geon.util.utils import execute
from geon.util.utils import in_bound_environment
from geon.util.utils import transform_numeric_derivative
from geon.util.utils import transform_derivative
import geon as be
import geon.frontends.base.axis as ax


delta = 1e-3
rtol = atol = 1e-2


@in_bound_environment
def test_expand_dims():
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
                for axis, length in zip(tensor_axes, tensor_axes_lengths):
                    axis.length = length

                new_axis = test['new_axis']
                new_axis.length = new_axis_length

                tensor_np = np.array(
                    test['tensor'], dtype=np.float32
                )
                tensor = be.placeholder(axes=be.Axes(tensor_axes))
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


@in_bound_environment
def test_slice():
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
            'sliced_axes': (ax.N,),
            'axes_lengths': {ax.C: 2, ax.D: 3, ax.N: 2},
            'expected': [5, 6]
        }
    ]

    for test in tests:
        for axis, length in test['axes_lengths'].items():
            axis.length = length
        tensor_axes = test['tensor_axes']

        tensor_np = np.array(
            test['tensor'], dtype='float32'
        )
        tensor = be.placeholder(axes=be.Axes(tensor_axes))
        tensor.value = tensor_np
        expected = np.array(test['expected'], dtype='float32')

        s = test['slice']
        s_axes = test['sliced_axes']

        sliced = be.Slice(tensor, s, s_axes)
        sliced_val, = execute([sliced])

        assert np.array_equal(sliced_val, expected)

        # Test backpropagation
        numeric_deriv = transform_numeric_derivative(
            sliced, tensor, delta
        )
        sym_deriv = transform_derivative(sliced, tensor)

        assert np.allclose(
            numeric_deriv, sym_deriv, rtol=rtol, atol=atol
        )


@in_bound_environment
def test_padding():
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
            'padded_axes': (ax.M, ax.N),
            'axes_lengths': {ax.C: 2, ax.D: 3, ax.M: 3, ax.N: 5}
        }
    ]

    for test in tests:
        for axis, length in test['axes_lengths'].items():
            axis.length = length
        tensor_axes = test['tensor_axes']
        tensor_np = np.array(
            test['tensor'], dtype='float32'
        )
        tensor = be.placeholder(axes=be.Axes(tensor_axes))
        tensor.value = tensor_np

        padding = test['padding']
        np_padding = map(
            lambda p: (p, p) if isinstance(p, int) else p,
            test['padding']
        )
        padded_axes = test['padded_axes']
        padded = be.pad(tensor, padded_axes, padding)

        expected_val = np.pad(tensor_np, np_padding, mode='constant')
        computed_val, = execute([padded])

        assert np.array_equal(expected_val, computed_val)

        # Test backpropagation
        numeric_deriv = transform_numeric_derivative(
            padded, tensor, delta
        )
        sym_deriv = transform_derivative(padded, tensor)

        assert np.allclose(
            numeric_deriv, sym_deriv, rtol=rtol, atol=atol
        )
