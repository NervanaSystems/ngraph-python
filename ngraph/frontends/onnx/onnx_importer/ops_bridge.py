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
from ngraph.frontends.onnx.onnx_importer.utils import verify_axes_binary_broadcast_compatible, \
    make_reduction_op, cast_axes_for_matmul
from ngraph.frontends.tensorflow.tf_importer.utils_pos_axes import cast_to_pos_axes

logger = logging.getLogger(__name__)


class OpsBridge:
    """
    Bridging ops between ONNX and ngraph.
    """

    def get_ng_node(self, onnx_node):  # type: (NodeWrapper) -> Op
        """
        Create an ngraph Op from an ONNX node definition.
        """
        op_type = onnx_node.op_type
        ng_node_factory = getattr(self, op_type, None)
        ng_inputs = onnx_node.get_ng_inputs()

        if not ng_node_factory:
            raise NotImplementedError("Unknown operation: %s", op_type)

        return ng_node_factory(onnx_node, ng_inputs)

    # Unary Ops
    def Abs(self, onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
        return ng.absolute(ng_inputs[0])

    def Relu(self, onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
        return ng.maximum(ng_inputs[0], 0.)

    # Reduction Ops
    def ReduceSum(self, onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
        return make_reduction_op(ng.sum, onnx_node, ng_inputs[0])

    def ReduceMax(self, onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
        return make_reduction_op(ng.max, onnx_node, ng_inputs[0])

    def ReduceMin(self, onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
        return make_reduction_op(ng.min, onnx_node, ng_inputs[0])

    def ReduceLogSumExp(self, onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
        op = ng.exp(ng_inputs[0])
        op = make_reduction_op(ng.sum, onnx_node, op)
        op = ng.log(op)
        return op

    def ReduceMean(self, onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
        return make_reduction_op(ng.mean, onnx_node, ng_inputs[0])

    def ReduceProd(self, onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
        return make_reduction_op(ng.prod, onnx_node, ng_inputs[0])

    # Binary Ops
    def Add(self, onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
        verify_axes_binary_broadcast_compatible(onnx_node, ng_inputs)
        return ng.add(ng_inputs[0], ng_inputs[1])

    def Sub(self, onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
        verify_axes_binary_broadcast_compatible(onnx_node, ng_inputs)
        return ng.subtract(ng_inputs[0], ng_inputs[1])

    def Mul(self, onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
        verify_axes_binary_broadcast_compatible(onnx_node, ng_inputs)
        return ng.multiply(ng_inputs[0], ng_inputs[1])

    def Div(self, onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
        verify_axes_binary_broadcast_compatible(onnx_node, ng_inputs)
        return ng.divide(ng_inputs[0], ng_inputs[1])

    # Matrix multiplication
    def Dot(self, onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
        left, right = cast_axes_for_matmul(*ng_inputs)
        return cast_to_pos_axes(ng.dot(left, right))

    def Gemm(self, onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
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

    # Misc
    def Constant(self, onnx_node, ng_inputs):  # type: (NodeWrapper, List[TensorOp]) -> Op
        value_tensor = onnx_node.get_attribute_value('value')
        return cast_to_pos_axes(ng.constant(value_tensor.to_array()))
