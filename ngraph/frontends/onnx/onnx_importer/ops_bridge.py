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

import ngraph as ng
from ngraph.frontends.onnx.onnx_importer.utils.reduction import make_reduction_op
from ngraph.frontends.onnx.onnx_importer.utils.binary import \
    verify_axes_binary_broadcast_compatible, cast_axes_for_matmul
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
    verify_axes_binary_broadcast_compatible(onnx_node, ng_inputs)
    return ng.add(ng_inputs[0], ng_inputs[1])


def Sub(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    verify_axes_binary_broadcast_compatible(onnx_node, ng_inputs)
    return ng.subtract(ng_inputs[0], ng_inputs[1])


def Mul(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    verify_axes_binary_broadcast_compatible(onnx_node, ng_inputs)
    return ng.multiply(ng_inputs[0], ng_inputs[1])


def Div(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    verify_axes_binary_broadcast_compatible(onnx_node, ng_inputs)
    return ng.divide(ng_inputs[0], ng_inputs[1])


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


# Misc
def Constant(onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
    value_tensor = onnx_node.get_attribute_value('value')
    return cast_to_pos_axes(ng.constant(value_tensor.to_array()))
