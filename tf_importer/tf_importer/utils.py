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

import ngraph as ng


def tensor_shape_to_tuple(tf_shape):
    """
    Convert tensorflow's tensor shape to tuple

    Args:
        tf_shape: TensorShape object

    Returns:
        tuple of the shape
    """
    return tuple([s.value for s in tf_shape])


def tf_shape_to_axes(tf_shape):
    """
    Convert tensorflow's tensor shape to ngraph axes

    Arguments:
        tf_shape: attr_value_pb2.AttrValue, tf node's shape
    Returns:
        ng.Axes
    """
    shape = [int(d.size) for d in tf_shape.shape.dim]
    axes = [ng.Axis(s) for s in shape]
    return tuple(axes)


def shape_to_axes(shape):
    """
    Convert shape to axes

    Args:
        shape: shape of tensor

    Returns:
        ng.Axes object
    """
    if not shape:
        return ng.Axes()
    axes = [ng.Axis(length=s) for s in shape]
    return axes


def is_compatible_numpy_shape(left_shape, right_shape):
    """
    Check if left_shape and right_shape are numpy-compatible

    Args:
        left_shape: shape of the left tensor
        right_shape: shale of the right tensor

    Returns:
        True if numpy-compatible, False otherwise.
    """
    if (not left_shape) or (not right_shape):
        return True
    for l, r in zip(reversed(left_shape), reversed(right_shape)):
        if l == 1 or r == 1:
            continue
        elif l != r:
            return False
    return True
