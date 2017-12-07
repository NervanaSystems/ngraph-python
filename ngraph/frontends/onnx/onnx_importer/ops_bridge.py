# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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

from __future__ import print_function
from __future__ import division

import logging
from string import ascii_letters

import ngraph as ng
from ngraph.frontends.onnx.onnx_importer.utils.axes import reorder_axes, reshape_workaround
from ngraph.frontends.onnx.onnx_importer.utils.misc import split_into_pairs
from ngraph.frontends.onnx.onnx_importer.utils.pool import make_pooling_op, make_global_pooling_op
from ngraph.frontends.onnx.onnx_importer.utils.reduction import make_reduction_op
from ngraph.frontends.onnx.onnx_importer.utils.binary import cast_axes_for_binary_broadcast, \
    cast_axes_for_matmul
from ngraph.frontends.onnx.onnx_importer.utils.conv import make_convolution_op
from ngraph.frontends.tensorflow.tf_importer.utils_pos_axes import cast_to_pos_axes

logger = logging.getLogger(__name__)


def make_ng_node(onnx_node):  # type: (NodeWrapper) -> Op
    """
    Create an ngraph Op from an ONNX node definition.
    """
    op_type = onnx_node.op_type

    try:
        ng_node_factory = globals()[op_type]
    except KeyError:
        raise NotImplementedError("Unknown operation: %s", op_type)

    ng_inputs = onnx_node.get_ng_inputs()
    return ng_node_factory(onnx_node, ng_inputs)


# Unary Ops
def Abs(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    return ng.absolute(ng_inputs[0])


def Sigmoid(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    return ng.sigmoid(ng_inputs[0])


def Tanh(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    return ng.tanh(ng_inputs[0])


def Relu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    return ng.maximum(ng_inputs[0], 0.)


def LeakyRelu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    alpha = onnx_node.get_attribute_value('alpha', 0.01)
    if not 0 <= alpha <= 1:
        logger.warning('LeakyRelu node (%s): alpha value should be in range (0,1), but is: %s',
                       onnx_node.name, alpha)

    return ng.maximum(alpha * ng_inputs[0], 0.)


def PRelu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    slope = onnx_node.get_attribute_value('slope', 0.01)
    if not 0 <= slope <= 1:
        logger.warning('PRelu node (%s): slope value should be in range (0,1), but is: %s',
                       onnx_node.name, slope)

    return ng.maximum(slope * ng_inputs[0], ng_inputs[0])


def Selu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    # f(x) = gamma * (alpha * exp(x) - alpha) for x <= 0, f(x) = gamma * x for x > 0
    x = ng_inputs[0]
    alpha = onnx_node.get_attribute_value('alpha', 1.6732)
    gamma = onnx_node.get_attribute_value('gamma', 1.0507)

    return gamma * (ng.maximum(x, 0) + alpha * (ng.exp(-ng.maximum(-x, 0)) - 1))


def Elu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    # f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0
    x = ng_inputs[0]
    alpha = onnx_node.get_attribute_value('alpha', 1)

    if not alpha < 0:
        logger.warning('Elu node (%s): alpha value should be positive, but is: %s',
                       onnx_node.name, alpha)

    return ng.maximum(x, 0) + alpha * (ng.exp(-ng.maximum(-x, 0)) - 1)


# Reduction Ops
def ReduceSum(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    return make_reduction_op(ng.sum, onnx_node, ng_inputs[0])


def ReduceMax(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    return make_reduction_op(ng.max, onnx_node, ng_inputs[0])


def ReduceMin(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    return make_reduction_op(ng.min, onnx_node, ng_inputs[0])


def ReduceLogSumExp(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    op = ng.exp(ng_inputs[0])
    op = make_reduction_op(ng.sum, onnx_node, op)
    op = ng.log(op)
    return op


def ReduceMean(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    return make_reduction_op(ng.mean, onnx_node, ng_inputs[0])


def ReduceProd(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    return make_reduction_op(ng.prod, onnx_node, ng_inputs[0])


# Binary Ops
def Add(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    left, right = cast_axes_for_binary_broadcast(onnx_node, ng_inputs)
    return ng.add(left, right)


def Sub(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    left, right = cast_axes_for_binary_broadcast(onnx_node, ng_inputs)
    return ng.subtract(left, right)


def Mul(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    left, right = cast_axes_for_binary_broadcast(onnx_node, ng_inputs)
    return ng.multiply(left, right)


def Div(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    left, right = cast_axes_for_binary_broadcast(onnx_node, ng_inputs)
    return ng.divide(left, right)


# Matrix multiplication
def Dot(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    left, right = cast_axes_for_matmul(*ng_inputs)
    return cast_to_pos_axes(ng.dot(left, right))


def Gemm(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    # Y = alpha * (A @ B) + beta * C
    input_a, input_b, input_c = ng_inputs
    alpha = onnx_node.get_attribute_value('alpha', 1)  # Scalar multiplier for A @ B
    beta = onnx_node.get_attribute_value('beta', 1)  # Scalar multiplier for input tensor C
    broadcast = onnx_node.get_attribute_value('broadcast', 1)  # Should C be broadcast?
    trans_a = onnx_node.get_attribute_value('transA', False)  # Should A be transposed?
    trans_b = onnx_node.get_attribute_value('transB', False)  # Should B be transposed?

    if not broadcast:
        logger.warning('Gemm node (%s): import does not support broadcast value %s',
                       onnx_node.name, broadcast)

    if trans_a:
        input_a = ng.Transpose(input_a)

    if trans_b:
        input_b = ng.Transpose(input_b)

    input_a, input_b = cast_axes_for_matmul(input_a, input_b)
    a_dot_b = ng.dot(input_a, input_b)
    a_dot_b = cast_to_pos_axes(a_dot_b)
    return alpha * a_dot_b + beta * input_c


# Convolution ops
def Conv(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    return cast_to_pos_axes(make_convolution_op(onnx_node, ng_inputs))


def ConvTranspose(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    return cast_to_pos_axes(make_convolution_op(onnx_node, ng_inputs, transpose=True))


def Pad(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    paddings = onnx_node.get_attribute_value('paddings')
    constant = 'constant'
    mode = onnx_node.get_attribute_value('mode', constant)  # 'constant', 'reflect' or 'edge'
    value = onnx_node.get_attribute_value('value', 0)

    if mode != constant or value != 0:
        raise NotImplementedError('Pad node (%s): only constant padding with value=0 '
                                  'is supported.', onnx_node.name)

    # Split paddings into pairs for each axis
    paddings = [pad for pad in split_into_pairs(paddings)]
    return cast_to_pos_axes(ng.pad(ng_inputs[0], paddings))


# Pooling
def AveragePool(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    return cast_to_pos_axes(make_pooling_op(onnx_node, ng_inputs))


def MaxPool(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    return cast_to_pos_axes(make_pooling_op(onnx_node, ng_inputs))


def GlobalMaxPool(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    """Equivalent to MaxPool with kernel size equal to the spatial dimension of input tensor"""
    return cast_to_pos_axes(make_global_pooling_op(onnx_node, ng_inputs))


def GlobalAveragePool(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    """Equivalent to AveragePool with kernel size equal to the spatial dimension of input tensor"""
    return cast_to_pos_axes(make_global_pooling_op(onnx_node, ng_inputs))


# Reshape ops
def Flatten(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    """Flatten the input tensor into a 2D matrix"""
    data = ng_inputs[0]
    axis = onnx_node.get_attribute_value('axis', 1)

    if not (0 <= axis <= len(data.axes)):
        raise ValueError('Flatten node (%s): %d is not a valid value for `axis`.',
                         onnx_node.name, axis)

    return cast_to_pos_axes(ng.flatten_at(data, axis))


def Transpose(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    """Transpose the input tensor similar to numpy.transpose

    By default, reverse the dimensions, but if `perm` attribute is specified
    permute the axes according to the values given.
    """
    data = ng_inputs[0]
    permute_axes = onnx_node.get_attribute_value('perm')

    if permute_axes:
        input_template = ''.join([ascii_letters[i] for i in range(len(data.axes))])
        output_template = ''.join([ascii_letters[i] for i in permute_axes])
        ng_op = reorder_axes(data, input_template, output_template)
    else:
        ng_op = ng.Transpose(data)

    return cast_to_pos_axes(ng_op)


def Slice(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    """Produce a slice of the input tensor along multiple axes."""
    x = ng_inputs[0]

    starts = onnx_node.get_attribute_value('starts')
    ends = onnx_node.get_attribute_value('ends')
    if not (starts and ends and len(starts) == len(ends)):
        raise ValueError('Slice node (%s): attributes `starts` and `ends` must be set '
                         'and of equal length.', onnx_node.name)

    axes = onnx_node.get_attribute_value('axes', list(range(len(starts))))
    slices_count = max(len(axes), *starts)
    if slices_count > len(x.axes):
        raise ValueError('Slice node (%s): specifies %d slices, there are only %d input axes.',
                         onnx_node.name, slices_count, len(x.axes))

    slices = [slice(starts[axes.index(axis_number)], ends[axes.index(axis_number)])
              if (axis_number in axes) else slice(None) for axis_number in range(len(x.axes))]

    return cast_to_pos_axes(ng.tensor_slice(x, slices))


def Concat(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    """Concatenate a list of tensors into a single tensor"""
    axis = onnx_node.get_attribute_value('axis', 0)

    if len(ng_inputs) < 2:
        raise ValueError('Concat node (%s): requires at least 2 inputs, %d given.',
                         onnx_node.name, len(ng_inputs))

    unique_input_ranks = {len(node.axes) for node in ng_inputs}
    if len(unique_input_ranks) != 1:
        raise ValueError('Concat node (%s): input tensors must be of equal rank.', onnx_node.name)

    if axis >= unique_input_ranks.pop():
        raise ValueError('Concat node (%s): `axis` attribute is out of range.', onnx_node.name)

    ng_axis = ng_inputs[0].axes[axis]
    return ng.concat_along_axis(ng_inputs, ng_axis)


def Squeeze(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    """Remove single-dimensional entries from the shape of a tensor"""
    data = ng_inputs[0]
    axes_to_squeeze = onnx_node.get_attribute_value('axes')

    if max(axes_to_squeeze) >= len(data.axes):
        raise ValueError('Squeeze node (%s): `axes` attribute value %d is out of range.',
                         onnx_node.name, max(axes_to_squeeze))

    slices = [0 if index in axes_to_squeeze else
              slice(None) for index, axis in enumerate(data.axes)]

    return ng.tensor_slice(data, slices)


def Reshape(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    """Reshape the input tensor similar to numpy.reshape"""
    data = ng_inputs[0]
    shape = onnx_node.get_attribute_value('shape', data.axes.lengths)

    # This is code we want to use, but cannot due to a bug:
    # https://github.com/NervanaSystems/private-ngraph/issues/2372
    """
    new_axes = ng.make_axes([ng.make_axis(length=length) for length in shape])
    x = ng.flatten(data)
    x = ng.cast_axes(x, new_axes.flatten())
    x = ng.unflatten(x)
    return cast_to_pos_axes(x)
    """
    return reshape_workaround(data, shape)


# Misc
def Constant(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    value_tensor = onnx_node.get_attribute_value('value')
    return cast_to_pos_axes(ng.constant(value_tensor.to_array()))
