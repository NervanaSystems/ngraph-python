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
import ngraph as ng
from ngraph.frontends.tensorflow.tf_importer.utils_pos_axes import \
    make_pos_axes, cast_to_pos_axes
from ngraph.frontends.tensorflow.tf_importer.utils_broadcast import \
    broadcast_to, broadcasted_shape


# Placeholder op
def placeholder(dtype=None, shape=None, name=None):
    """
    TF's signature: placeholder(dtype, shape=None, name=None), here we match
    ngraph's signature more closely.
    """
    shape = () if shape is None else shape
    axes = make_pos_axes(shape)
    return ng.placeholder(axes, dtype=dtype).named(name)


# Binary element wise ops
def _element_wise_binary(x, y, ng_op, name=None):
    out_shape = broadcasted_shape(x.axes.lengths, y.axes.lengths)
    x = broadcast_to(x, out_shape)
    y = broadcast_to(y, out_shape)
    return ng_op(x, y).named(name)


def add(x, y, name=None):
    return _element_wise_binary(x, y, ng.add, name=name)


def subtract(x, y, name=None):
    return _element_wise_binary(x, -y, ng.add, name=name)


def multiply(x, y, name=None):
    return _element_wise_binary(x, y, ng.multiply, name=name)


def divide(x, y, name=None):
    return _element_wise_binary(x, y, ng.divide, name=name)


def mod(x, y, name=None):
    return _element_wise_binary(x, y, ng.mod, name=name)


def maximum(x, y, name=None):
    return _element_wise_binary(x, y, ng.maximum, name=name)


def minimum(x, y, name=None):
    return _element_wise_binary(x, y, ng.minimum, name=name)


# Unary element wise ops
def tanh(x, name=None):
    return ng.tanh(x).named(name)


def sigmoid(x, name=None):
    return ng.sigmoid(x).named(name)


def relu(x, name=None):
    return ng.maximum(x, 0.).named(name)


def log(x, name=None):
    return ng.log(x).named(name)


def negative(x, name=None):
    return ng.negative(x).named(name)


def square(x, name=None):
    return ng.square(x).named(name)


# Reduction ops
def _reduction(input_tensor, ng_op, axis=None, keep_dims=False, name=None):
    """
    Args:
        axis: int or list of ints
    """
    if keep_dims:
        raise NotImplementedError("ngraph only support keep_dims=True now.")

    if axis is None:
        ng_reduction_axes = input_tensor.axes
    else:
        try:
            iter(axis)
        except TypeError:
            axis = list(axis)
        ng_reduction_axes = ng.make_axes([input_tensor.axes[ind] for ind in
                                          axis])
    res = ng_op(input_tensor, reduction_axes=ng_reduction_axes)
    return cast_to_pos_axes(res).named(name)


def reduce_max(input_tensor, axis=None, keep_dims=False, name=None):
    return _reduction(input_tensor, ng.max, axis=axis, keep_dims=keep_dims,
                      name=name)


def reduce_mean(input_tensor, axis=None, keep_dims=False, name=None):
    return _reduction(input_tensor, ng.mean, axis=axis, keep_dims=keep_dims,
                      name=name)


def reduce_min(input_tensor, axis=None, keep_dims=False, name=None):
    return _reduction(input_tensor, ng.min, axis=axis, keep_dims=keep_dims,
                      name=name)


def reduce_prod(input_tensor, axis=None, keep_dims=False, name=None):
    return _reduction(input_tensor, ng.prod, axis=axis, keep_dims=keep_dims,
                      name=name)


def reduce_sum(input_tensor, axis=None, keep_dims=False, name=None):
    return _reduction(input_tensor, ng.sum, axis=axis, keep_dims=keep_dims,
                      name=name)


# Constant ops
def constant(value, dtype=None, shape=None, name='Const', verify_shape=False):
    """
    TODO: handle auto dtype from value
    """
    # Convert to numpy
    np_value = np.array(value)

    # Check shape if needed
    if shape is not None and shape != np_value.shape:
        if verify_shape:
            raise ValueError("Since veryfy_shape is True, value's shape {} "
                             "must be same as the specified shape {} "
                             .format(np_value.shape, shape))
        elif np_value.shape == ():
            # Automatic broadcasting of scalar constant
            np_value = np.empty(shape)
            np_value.fill(value)
        else:
            raise ValueError("value's shape {} must be same as the specified "
                             "shape {}, or value must be a scalar"
                             .format(np_value.shape, shape))
    return ng.constant(np_value,
                       axes=make_pos_axes(np_value.shape),
                       dtype=dtype).named(name)


def ones(shape, dtype=np.float32, name=None):
    return constant(1., dtype=dtype, shape=shape, name=name)


def zeros(shape, dtype=np.float32, name=None):
    return constant(zeros, dtype=dtype, shape=shape, name=name)


# Matmul op
def matmul(left, right, transpose_a=False, transpose_b=False, name=None):
    """
    Only support 2d matmul for now.
    """
    # Transpose
    if transpose_a:
        left = ng.Transpose(left)
    if transpose_b:
        right = ng.Transpose(right)

    # Check shape
    assert len(left.axes) == len(right.axes) == 2
    assert left.axes[1].length == right.axes[0].length

    # step 1: cast left (pos_1, pos_0), right (pos_1, pos_0) =>
    #              left (temp , pos_1), right (pos_1, pos_0)
    # step 2: perform left dot right, result
    #         (temp, pos_0)
    # step 3: cast back to (post_1, pos_0)
    left_temp_axes = ng.make_axes([ng.make_axis(left.axes[0].length),
                                   right.axes[0]])
    left = ng.cast_axes(left, axes=left_temp_axes)

    # Result op
    result_op = ng.dot(left, right).named(name)
    result_op = cast_to_pos_axes(result_op)

    # Return
    return result_op.named(name)


# Variables
def Variable(initial_value=None, name=None, dtype=None):
    # Same behavior as TF
    if initial_value is None:
        raise ValueError("initial_value must be specified.")

    # Get axes
    try:
        axes = make_pos_axes(initial_value.shape)
    except:
        raise NotImplementedError('Shape must be know prior to execution')

    return ng.variable(axes,
                       dtype=dtype,
                       initial_value=initial_value).named(name)


# Neural network ops
def sparse_softmax_cross_entropy_with_logits(labels=None, logits=None,
                                             name=None):
    """
    Computes softmax cross entropy. The inputs `logits` are unscaled log
    probabilities, and each row of `labels[i]` must be a valid distribution.

    Args:
        labels: of axis (N,) for (POS_0,)
        logits: of axis (N, Y) for (POS_1, POS_0)
        name: name of the ngraph op
    """
    # Check input dimension
    #         (    N,     Y),         (    N)
    # logits: (pos_1, pos_0), labels: (pos_0)
    try:
        assert len(logits.axes) == 2
        assert len(labels.axes) == 1
        assert logits.axes[0].length == labels.axes[0].length
    except:
        raise NotImplementedError("logits' shape must be (N, Y), "
                                  "labels' shape must be (N,), "
                                  "other shapes not supported yet.")
    # get axis
    axis_n, axis_y = logits.axes

    # convert labels to one-hot labels
    labels = ng.cast_axes(labels, ng.make_axes(axis_n))
    labels = ng.one_hot(labels, axis=axis_y)
    labels = ng.axes_with_order(labels, axes=logits.axes)

    # predicts: (N, Y)
    predicts = ng.softmax(logits, normalization_axes=axis_y)

    # cross_entropy: (N)
    res = ng.cross_entropy_multi(predicts, labels, out_axes=(axis_n,))
    return cast_to_pos_axes(res).named(name)


def softmax(logits, dim=-1, name=None):
    if dim != -1:
        raise NotImplementedError("TODO: only support tf.nn.softmax(logits, "
                                  "dim=-1) now, should add more.")
    return ng.softmax(logits, normalization_axes=logits.axes[1])
