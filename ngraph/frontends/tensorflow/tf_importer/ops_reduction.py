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
import ngraph as ng
import numpy as np


class OpsReduction(OpsBase):
    """
    Mix-in class for reduction related ops
    """

    def _apply_reduction_op(self, reduction_op, tf_node, inputs):
        """
        Apply ngraph reduction op.

        Arguments:
            reduction_op: ngraph reduction Op.
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            The resulting ngraph node.
        """
        # get inputs
        input_tensor, reduction_indices = inputs

        # check reduction_indices comes from constant
        try:
            assert reduction_indices.const is not None
        except:
            raise NotImplementedError(
                "[NON-NATIVE] reduction_indices be "
                "constants, cannot come from intermediate "
                "results")

        # get reduction axes
        reduction_indices = [int(ind) for ind in reduction_indices.const]
        reduction_axes = ng.make_axes(
            [input_tensor.axes[ind] for ind in reduction_indices])

        # perform reduction operation
        result_op = reduction_op(input_tensor, reduction_axes=reduction_axes)

        # broadcast results for safety
        new_out_axes = [
            ng.make_axis(length=axis.length) for axis in result_op.axes
        ]
        result_op = ng.cast_axes(
            result_op, ng.make_axes(new_out_axes), name=tf_node.name)

        return result_op

    def Sum(self, tf_node, inputs):
        """
        Computes the sum of elements across dimensions of a tensor.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            input_tensor, reduction_indices, keep_dims, name
        """
        return self._apply_reduction_op(ng.sum, tf_node, inputs)

    def Mean(self, tf_node, inputs):
        """
        Computes the mean of elements across dimensions of a tensor.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            input_tensor, reduction_indices, keep_dims, name
        """
        return self._apply_reduction_op(ng.mean, tf_node, inputs)

    def Prod(self, tf_node, inputs):
        """
        Computes the product of elements across dimensions of a tensor.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            input_tensor, reduction_indices, keep_dims, name
        """
        # TODO: currently don't have native support in ngraph, can only handle
        #       static case

        # get inputs
        input_tensor, reduction_indices = inputs
        reduction_indices = [int(ind) for ind in reduction_indices.const]

        # can only handle constant case for now
        try:
            # check is constant
            assert input_tensor.const is not None

            # compute output numpy result
            input_np_tensor = input_tensor.const
            np_result = np.prod(input_np_tensor, axis=tuple(reduction_indices))

            # get output axis
            out_axes = [
                ng.make_axis(input_tensor.axes[i].length)
                for i in range(len(input_tensor.axes))
                if i not in reduction_indices
            ]

            # broadcast for safety
            result_op = ng.constant(np_result, axes=out_axes)
            return result_op
        except:
            raise NotImplementedError("reduce_prod currently not supported in "
                                      "ngraph, can only handle constant case.")
