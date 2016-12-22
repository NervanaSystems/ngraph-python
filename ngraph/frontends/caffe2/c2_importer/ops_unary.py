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

from ngraph.frontends.caffe2.c2_importer.ops_base import OpsBase
import ngraph as ng


class OpsUnary(OpsBase):
    """
    Mix-in class for unary ops
    """
    def _element_wise_unary(self, ng_op, c2_op, inputs):
        """
        Element-wise unary operation.

        Args:
            ng_op: ngraph Op to be applied.
            c2_op: NodeDef object, the caffe2 node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the caffe2 node.
        """
        # get inputs
        left = inputs[0]

        # result
        result_op = ng_op(left).named(c2_op.name)

        # return op
        return result_op

    def Tanh(self, c2_op, inputs):
        """
        Computes hyperbolic tangent of `x` element-wise.

        Arguments:
            c2_op: NodeDef object, the caffe2 node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the caffe2 node.

        Inputs to tf_node:
            x, name
        """
        return self._element_wise_unary(ng.tanh, c2_op, inputs).named(c2_op.name)

    def Relu(self, c2_op, inputs):
        """
        Computes rectified linear: `max(features, 0)`.

        Arguments:
            c2_op: NodeDef object, the caffe2 node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the caffe2 node.

        Inputs to tf_node:
            features, name
        """
        return ng.maximum(inputs[0], 0.).named(c2_op.name)

    def Softmax(self, c2_op, inputs):
        """
        Computes softmax: `exp(x)/sum(exp(x)`.

        Arguments:
            c2_op: NodeDef object, the caffe2 node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the caffe2 node.
        """
        # get input
        x = inputs[0]
        # normalization axes
        norm_axes = x.axes[1]

        return ng.softmax(x, norm_axes).named(c2_op.name)
