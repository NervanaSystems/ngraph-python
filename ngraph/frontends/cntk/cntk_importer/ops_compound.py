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


class OpsCompound:
    """
    Bridging compoud operations between CNTK and ngraph.
    """

    def cast_axes_for_compound_op(self, inputs):
        left, right = inputs

        left_dim = len(left.axes)
        right_dim = len(right.axes)

        # pad left and right axis to be the same length, align right
        result_dim = max(left_dim, right_dim)
        left_axes_pad = [
            ng.make_axis(length=1) for _ in range(result_dim - left_dim)
        ] + list(left.axes)
        right_axes_pad = [
            ng.make_axis(length=1) for _ in range(result_dim - right_dim)
        ] + list(right.axes)
        result_axes = [
            ng.make_axis(length=max(l.length, r.length))
            for l, r in zip(left_axes_pad, right_axes_pad)
        ]

        # broadcast left / right, introducing dummy length 1 axes
        left = ng.broadcast(left, left_axes_pad)
        right = ng.broadcast(right, right_axes_pad)

        # make two-way map of lr matching axes and map for result axes
        lr_axes_map = dict()
        result_axes_map = dict()
        for l, r, re in zip(left.axes, right.axes, result_axes):
            lr_axes_map[l] = r
            lr_axes_map[r] = l
            result_axes_map[l] = re
            result_axes_map[r] = re

        # get left / right slice
        left_slice = []
        right_slice = []
        for l, r in zip(left.axes, right.axes):
            if l.length == 1 and r.length != 1:
                left_slice.append(0)
            else:
                left_slice.append(slice(None))
            if r.length == 1 and l.length != 1:
                right_slice.append(0)
            else:
                right_slice.append(slice(None))

        # perform slicing
        left_sliced = ng.tensor_slice(left, left_slice)
        right_sliced = ng.tensor_slice(right, right_slice)

        # now cast the right_sliced to left_sliced from the axis map
        right_casted_axes = []
        for r in right_sliced.axes:
            if r in lr_axes_map and lr_axes_map[r] in left_sliced.axes:
                right_casted_axes.append(lr_axes_map[r])
            else:
                right_casted_axes.append(r)
        right_sliced_casted = ng.cast_axes(right_sliced, right_casted_axes)

        return left_sliced, right_sliced_casted

    def CrossEntropyWithSoftmax(self, cntk_op, inputs):
        """
        Computes the softmax cross entropy between the inputs[0] and inputs[1].

        Arguments:
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        cast_0, cast_1 = self.cast_axes_for_compound_op(inputs)

        if isinstance(cast_0, ng.AssignableTensorOp):
            cast_1 = ng.softmax(cast_1)
        else:
            cast_0 = ng.softmax(cast_0)

        return ng.cross_entropy_multi(cast_0, cast_1).named(cntk_op.uid)

    def Combine(self, cntk_op, inputs):
        """
        Returns combined outputs of inputs list.

        Arguments:
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        return ng.stack(inputs, ng.make_axis(len(inputs)))
