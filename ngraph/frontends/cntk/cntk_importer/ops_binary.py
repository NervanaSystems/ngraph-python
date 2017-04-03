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
from ngraph.frontends.tensorflow.tf_importer.ops_binary import OpsBinary as TFOpsBinary


class OpsBinary(TFOpsBinary):
    """
    Bridging binary operations between CNTK and ngraph.
    """

    def Plus(self, cntk_op, inputs):
        """
        Returns input[0] + input[1] element-wise.

        Arguments:
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        return self._element_wise_binary(ng.Add, cntk_op, inputs)

    def Minus(self, cntk_op, inputs):
        """
        Returns input[0] - input[1] element-wise.

        Arguments:
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        return self._element_wise_binary(ng.Subtract, cntk_op, inputs)

    def ElementTimes(self, cntk_op, inputs):
        """
        Returns input[0] x input[1] (element wise).

        Arguments:
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        return self._element_wise_binary(ng.Multiply, cntk_op, inputs)

    def Times(self, cntk_op, inputs):
        """
        Returns input[0] x input[1] (matrix multiplication).

        Arguments:
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        cast_0, cast_1 = inputs

        if len(cast_0.axes) == 1 & len(cast_1.axes) == 1:
            pass
        elif len(cast_0.axes) == 1:
            temp = next((x for x in cast_1.axes if x.length == 1), None)
            if temp is None:
                temp = ng.make_axis(1)
            cast_0 = ng.broadcast(cast_0, [temp, cast_0.axes])
        elif len(cast_1.axes) == 1:
            temp = next((x for x in cast_0.axes if x.length == 1), None)
            if temp is None:
                temp = ng.make_axis(1)
            cast_1 = ng.broadcast(cast_1, [ng.make_axis(1), cast_1.axes])

        cast_0 = ng.cast_axes(cast_0, [cast_0.axes[0], cast_1.axes[0]])
        return ng.dot(cast_0, cast_1).named(cntk_op.uid)
