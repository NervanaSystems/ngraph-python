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
import numpy as np
import tensorflow as tf


class SGDOptimizer(object):
    def __init__(self, lrate=0.1):
        self.lrate = lrate

    def minimize(self, cost):
        variables = list(cost.variables())
        grads = [ng.deriv(cost, variable) for variable in variables]
        with ng.Op.saved_user_deps():
            param_updates = [ng.assign(variable, variable - self.lrate * grad)
                             for variable, grad in zip(variables, grads)]
            updates = ng.doall(param_updates)
        return updates


def tf_to_shape_tuple(input):
    """
    Convert tf objects to shape tuple
    Args:
        input: tf.TensorShape, tf.Tensor, tf.AttrValue or tf.NodeDef
               the corresponding tensorflow object

    Returns:
        tuple: shape of the tensorflow object
    """
    if isinstance(input, tf.TensorShape):
        return tuple([int(i.value) for i in input])
    elif isinstance(input, tf.Tensor):
        return tf_to_shape_tuple(input.get_shape())
    elif isinstance(input, tf.AttrValue):
        return tuple([int(d.size) for d in input.shape.dim])
    elif isinstance(input, tf.NodeDef):
        return tf_to_shape_tuple(input.attr['shape'])
    else:
        raise TypeError("Input to `tf_to_shape_tuple` has the wrong type.")


def tf_to_shape_axes(input):
    """
    Convert tf objects to axes
    Args:
        input: tf.TensorShape, tf.Tensor, tf.AttrValue or tf.NodeDef
               the corresponding tensorflow object

    Returns:
        tuple: new axes of the tensorflow object
    """
    return shape_to_axes(tf_to_shape_tuple(input))


def shape_to_axes(shape):
    """
    Convert shape to axes

    Args:
        shape: shape of tensor

    Returns:
        Axes: Axes for shape.
    """
    if not shape:
        return ng.make_axes()
    axes = [ng.make_axis(length=s) for s in shape]
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


def to_int(input):
    """
    Convert np array, tuple, list or const value to int

    Args:
        input: tuple, list or const value
    Return:
        tuple, list or const of int(s)
    """
    if isinstance(input, np.ndarray):
        return input.astype(int)
    elif isinstance(input, tuple):
        return tuple([int(i) for i in input])
    elif isinstance(input, list):
        return [int(i) for i in input]
    else:
        return int(input)
