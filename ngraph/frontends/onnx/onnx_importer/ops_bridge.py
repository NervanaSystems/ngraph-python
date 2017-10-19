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

import ngraph as ng
from ngraph.frontends.onnx.onnx_importer.utils import verify_axes_binary_broadcast_compatible, \
    make_reduction_op


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
