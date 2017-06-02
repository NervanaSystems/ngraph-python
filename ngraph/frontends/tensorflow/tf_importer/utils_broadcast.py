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

"""
Utilities related to broadcasting
"""
import ngraph as ng
from ngraph.frontends.tensorflow.tf_importer.utils_pos_axes import \
    make_pos_axes, reorder_pos_axes


def is_compatible_numpy_shape(left_shape, right_shape):
    """
    Check if left_shape and right_shape are numpy-compatible.

    Arguments:
        left_shape: Shape of the left tensor.
        right_shape: Shape of the right tensor.

    Returns:
        True if numpy-compatible, False otherwise.
    """
    if len(left_shape) == 0 or len(right_shape) == 0:
        return True
    is_compatible = lambda l, r: l == 1 or r == 1 or l == r
    shorter_len = min(len(left_shape), len(right_shape))
    for l, r in zip(left_shape[-shorter_len:], right_shape[-shorter_len:]):
        if not is_compatible(l, r):
            return False
    return True


def is_compatible_broadcast_shape(src_shape, dst_shape):
    """
    Check if src_shape can be broadcasted to dst_shape
    dst_shape needs to be numpy compatible with src_shape, and each matching
    dimension for dst_shape must be equal or larger than src_shape

    Args:
        src_shape: tuple or list, shape of the source tensor
        dst_shape: tuple or list, shape of the desination tensor

    Returns:
        True if src_shape can be broadcasted to dst_shape
    """
    if len(src_shape) > len(dst_shape):
        return False
    is_compatible = lambda l, r: l == 1 or l == r
    for l, r in zip(src_shape, dst_shape[-len(src_shape):]):
        if not is_compatible(l, r):
            return False
    return True


def broadcasted_shape(left_shape, right_shape):
    """
    Get broacasted shape of binary elementwise operation on shape_x and shape_y.
    For example, broadcasted_shape((1, 3), (2, 4, 1)) = (2, 4, 3)

    Args:
        left_shape: shape of LHS operand
        right_shape: shape of RHS operand

    Returns:
        The broadcasted shape.
    """
    if not is_compatible_numpy_shape(left_shape, right_shape):
        raise ValueError("left_shape {} and right_shape {} is not compatible."
                         .format(left_shape, right_shape))

    # pad to same length
    left_shape, right_shape = tuple(left_shape), tuple(right_shape)
    length_diff = abs(len(left_shape) - len(right_shape))
    if len(left_shape) > len(right_shape):
        right_shape = (0,) * length_diff + right_shape
    else:
        left_shape = (0,) * length_diff + left_shape

    # get broadcasted shape
    out_shape = [max(l, r) for l, r in zip(left_shape, right_shape)]
    return tuple(out_shape)


def broadcast_to(x, out_shape):
    """
    Broadcast tensor x to out_shape.

    Args:
        x: tensor to be broadcasted
        out_shape: tuple of the targeted shape

    Example:

         [3][2][1][0]    [4][3][2][1][0]
    from (5, 1, 2, 1) to (4, 5, 1, 2, 3)

    # step 1:
                                        [3][2][1]
    collapse 1 that will be broadcasted (5, 1, 2)

    # step 2:
             [4][0]          [4][0][3][2][1]
    add with (4, 3), becomes (4, 3, 5, 1, 2)

    # step 3:
               [4][3][2][1][0]
    reorder to (4, 5, 1, 2, 3)

    Returns:
        x broadcasted to outs_shape
    """
    if not is_compatible_broadcast_shape(x.axes.lengths, out_shape):
        raise ValueError("x's shape {} is not broadcastable to out_shape {}"
                         .format(x.axes.lengths, out_shape))
    x_ndims = len(x.axes)

    if x_ndims == 0:
        # special case: x'shape is same as out_shape
        return x

    elif x.axes.lengths == out_shape:
        # special case: scalar
        zero = ng.constant(0., axes=make_pos_axes(out_shape))
        return x + zero

    else:
        # collapse (collapse all dimension 1 axes that will be broadcasted)
        x_slice = []
        sliced_indices = []
        for index, (x_len, out_len) in enumerate(zip(x.axes.lengths,
                                                     out_shape[-len(x.axes):])):
            if x_len == 1 and out_len != 1:
                x_slice.append(0)
                sliced_indices.append(index)
            else:
                x_slice.append(slice(None))
        x = ng.tensor_slice(x, x_slice)

        # get the axes for the dummy zero
        zero_positions = [x_ndims - i - 1 for i in sliced_indices]
        zero_positions += list(range(x_ndims, len(out_shape)))
        zero_shape = [out_shape[-i - 1] for i in zero_positions]
        zero = ng.constant(0., axes=make_pos_axes(zero_shape,
                                                  positions=zero_positions))

        # broadcast and reorder
        x = reorder_pos_axes(x + zero)

        return x
