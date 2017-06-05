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
    is_compatible_numpy_shape
import ngraph as ng


class LegacyOpsBinary():
    def _element_wise_binary(self, ng_op, inputs, name=None):
        """
        Element-wise binary operation with broadcast.
        Args:
            ng_op: ngraph Op to be applied.
            inputs: List of ngraph Ops as inputs to this node.
            name: name of the ngraph op
        Returns:
            A ngraph Op corresponding to the element-wise binary op
        """
        # get inputs
        left, right = inputs

        # check if shape compatibility
        left_shape = left.axes.lengths
        right_shape = right.axes.lengths
        assert is_compatible_numpy_shape(left_shape, right_shape)

        if left_shape and right_shape and left_shape != right_shape:
            """
            Cast axes in numpy broadcast mapping rule
            1. introduce dummy length 1 axes to match left / right length
            2. keep maps for matching left / right / result axes
            3. slice left / right to remove length 1 axes if not both of them
               are length 1
            4. cast right to left by matching axes
            5. perform binary op
            6. cast and broadcast result
            """

            left_dim = len(left.axes)
            right_dim = len(right.axes)

            # pad left and right axis to be the same length, align right
            result_dim = max(left_dim, right_dim)
            left_axes_pad = [ng.make_axis(length=1) for _ in
                             range(result_dim - left_dim)] + list(left.axes)
            right_axes_pad = [ng.make_axis(length=1) for _ in
                              range(result_dim - right_dim)] + list(right.axes)
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

            # perform binary op
            result_op = ng_op(left_sliced, right_sliced_casted)

            # cast result axis and broadcast to full result axes
            trimmed_result_axes = [
                result_axes_map[re] for re in result_op.axes
            ]
            result_op = ng.cast_axes(result_op, trimmed_result_axes)
            result_op = ng.axes_with_order(result_op, axes=result_axes)

        elif left_shape == right_shape:
            # cast right axes to be the same as left
            right = ng.cast_axes(right, left.axes)
            result_op = ng_op(left, right).named(name)

        else:
            # no need for casting
            result_op = ng_op(left, right).named(name)

        # return op
        return result_op
