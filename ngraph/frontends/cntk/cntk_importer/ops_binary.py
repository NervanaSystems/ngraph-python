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
from ngraph.frontends.common.utils import squeeze_axes


class OpsBinary:
    """
    Bridging binary operations between CNTK and ngraph.
    """

    def _match_axes(self, bigger, smaller):
        """
        Returns set of axes for bigger input with axes
        matching axes in smaller input.

        Arguments:
            bigger: Input with more axes.
            smaller: Input with fewer axes.

        Returns:
            List of axes to be casted.
        """
        axes = []
        for i, axis_0 in enumerate(bigger):
            for axis_1 in smaller:
                if axis_0.length == axis_1.length:
                    if axis_1 not in axes and axis_0.name != 'N':
                        axes.append(axis_1)
                        break
            if len(axes) == i:
                axes.append(axis_0)
        return axes

    def _cast_for_binary_op(self, inputs):
        """
        Cast axes for input with more axes by matching
        its axes with second input's axes.

        Arguments:
            inputs: List of inputs to be casted.

        Returns:
            Casted inputs.
        """
        assert len(inputs) == 2

        cast_0, cast_1 = squeeze_axes(inputs)

        if len(cast_0.axes) >= len(cast_1.axes):
            axes = self._match_axes(cast_0.axes, cast_1.axes)
            cast_0 = ng.cast_axes(cast_0, axes)
        else:
            axes = self._match_axes(cast_1.axes, cast_0.axes)
            cast_1 = ng.cast_axes(cast_1, axes)

        return cast_0, cast_1

    def Plus(self, cntk_op, inputs):
        """
        Returns input[0] + input[1] element-wise.

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 2

        cast_0, cast_1 = self._cast_for_binary_op(inputs)
        return ng.add(cast_0, cast_1).named(cntk_op.uid)

    def Minus(self, cntk_op, inputs):
        """
        Returns input[0] - input[1] element-wise.

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 2

        cast_0, cast_1 = self._cast_for_binary_op(inputs)
        return ng.subtract(cast_0, cast_1).named(cntk_op.uid)

    def ElementTimes(self, cntk_op, inputs):
        """
        Returns input[0] x input[1] (element wise).

        Arguments:
             cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 2

        cast_0, cast_1 = self._cast_for_binary_op(inputs)
        return ng.multiply(cast_0, cast_1).named(cntk_op.uid)

    def Times(self, cntk_op, inputs):
        """
        Returns input[0] x input[1] (matrix multiplication).

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 2

        cast_0, cast_1 = inputs

        cast_0_len = len(cast_0.axes)
        cast_1_len = len(cast_1.axes)

        if cast_0_len == cast_1_len == 1:
            if cast_0.axes[0] != cast_1.axes[0]:
                cast_0 = ng.cast_axes(cast_0, cast_1.axes)
        elif cast_0_len == cast_1_len:
            if cast_0.axes[1].length == cast_1.axes[0].length:
                axes = [cast_0.axes[0], cast_1.axes[0]]
                axes.extend(cast_0.axes[2::])
                cast_0 = ng.cast_axes(cast_0, axes=axes)
            else:
                axes = self._match_axes(cast_0.axes, cast_1.axes)
                cast_0 = ng.cast_axes(cast_0, axes=axes)
        elif cast_0_len > cast_1_len:
            axes = self._match_axes(cast_0.axes, cast_1.axes)
            cast_0 = ng.cast_axes(cast_0, axes=axes)
        else:
            axes = self._match_axes(cast_1.axes, cast_0.axes)
            cast_1 = ng.cast_axes(cast_1, axes=axes)

        return ng.dot(cast_0, cast_1).named(cntk_op.uid)
