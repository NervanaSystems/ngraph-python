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

from ngraph.frontends.tensorflow.tf_importer.ops_base import OpsBase
from ngraph.frontends.tensorflow.tf_importer.utils_pos_axes import \
    cast_to_pos_axes
import ngraph as ng


class OpsMatmul(OpsBase):
    """
    Mix-in class for matrix multiplication ops:
    """

    def MatMul(self, tf_node, inputs):
        """
        Multiplies matrix `a` by matrix `b`. The inputs must be two-dimensional,
        the inner dimensions must match (possibly after transpose).

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            a, b, transpose_a, transpose_b, a_is_sparse, b_is_sparse, name
        """
        # get inputs
        left, right = inputs
        if tf_node.attr['transpose_a'].b:
            left = ng.Transpose(left)
        if tf_node.attr['transpose_b'].b:
            right = ng.Transpose(right)

        # check shape
        assert len(left.axes) == len(right.axes) == 2
        assert left.axes[1].length == right.axes[0].length

        # step 1: cast left (pos_1, pos_0), right (pos_1, pos_0) =>
        #              left (temp , pos_1), right (pos_1, pos_0)
        # step 2: perform left dot right, result
        #         (temp, pos_0)
        # step 3: cast back to (post_1, pos_0)
        left_temp_axes = ng.make_axes([ng.make_axis(left.axes[0].length),
                                       right.axes[0]])
        left = ng.cast_axes(left, axes=left_temp_axes)

        # result op
        result_op = ng.dot(left, right).named(tf_node.name)
        result_op = cast_to_pos_axes(result_op)

        # return
        return result_op
