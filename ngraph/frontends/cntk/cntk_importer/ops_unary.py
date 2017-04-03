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

import ngraph as ng


class OpsUnary:
    """
    Bridging unary operations between CNTK and ngraph.
    """

    def Sigmoid(self, cntk_op, inputs):
        """
        Returns element-wise sigmoid of inputs[0].

        Arguments:
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        return ng.sigmoid(inputs[0]).named(cntk_op.uid)

    def Exp(self, cntk_op, inputs):
        """
        Returns element-wise exp of inputs[0].

        Arguments:
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        return ng.exp(inputs[0]).named(cntk_op.uid)

    def Tanh(self, cntk_op, inputs):
        """
        Returns element-wise tanh of inputs[0].

        Arguments:
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        return ng.tanh(inputs[0]).named(cntk_op.uid)

    def Reciprocal(self, cntk_op, inputs):
        """
        Returns element-wise reciprocal of inputs[0].

        Arguments:
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        return ng.reciprocal(inputs[0]).named(cntk_op.uid)

    def ReLU(self, cntk_op, inputs):
        """
        Returns element-wise rectified linear of inputs[0].

        Arguments:
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        return ng.maximum(inputs[0], 0.).named(cntk_op.uid)
