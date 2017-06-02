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


from ngraph.frontends.tensorflow.tf_importer.utils_broadcast import \
    broadcast_to, broadcasted_shape
from ngraph.frontends.tensorflow.tf_importer.ops_base import OpsBase
import ngraph as ng


class OpsBinary(OpsBase):
    """
    Mix-in class element-wise binary ops.
    """

    def Add(self, tf_node, inputs):
        """
        Returns x + y element-wise. Supports broadcasting.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            x, y, name
        """
        return self._element_wise_binary(ng.add, tf_node, inputs)

    def BiasAdd(self, tf_node, inputs):
        """
        Special case for add where bias is 1d, and bias's length is equal to
        value's last dim.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            value, bias
        """
        value, bias = inputs
        if len(bias.axes) != 1:
            raise ValueError("Bias's must be 1D.")
        if bias.axes.lengths[0] != value.axes.lengths[-1]:
            raise ValueError("Bias's length must equal to value's last dim.")
        return self.Add(tf_node, inputs)

    def Sub(self, tf_node, inputs):
        """
        Returns x - y element-wise. Supports broadcasting.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            x, y, name
        """
        return self._element_wise_binary(ng.add, tf_node,
                                         [inputs[0], -inputs[1]])

    def Mul(self, tf_node, inputs):
        """
        Returns x * y element-wise. Supports broadcasting.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            x, y, name
        """
        return self._element_wise_binary(ng.multiply, tf_node, inputs)

    def Div(self, tf_node, inputs):
        """
        Returns x / y element-wise. Supports broadcasting.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            x, y, name
        """
        return self._element_wise_binary(ng.divide, tf_node, inputs)

    def Mod(self, tf_node, inputs):
        """
        Returns x mod y element-wise. Supports broadcasting.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            x, y, name
        """
        return self._element_wise_binary(ng.mod, tf_node, inputs)

    def Maximum(self, tf_node, inputs):
        """
        Returns element-wise maximum. Supports broadcasting.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            x, y, name
        """
        return self._element_wise_binary(ng.maximum, tf_node, inputs)

    def _element_wise_binary(self, ng_op, tf_node, inputs):
        """
        Element-wise binary operation with broadcast.

        Args:
            ng_op: ngraph Op to be applied.
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.
        """
        # get inputs
        left, right = inputs
        out_shape = broadcasted_shape(left.axes.lengths, right.axes.lengths)

        # broadcast left and right
        left = broadcast_to(left, out_shape)
        right = broadcast_to(right, out_shape)

        return ng_op(left, right).named(tf_node.name)
