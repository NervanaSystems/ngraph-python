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

from tf_importer.tf_importer.ops_base import OpsBase
import ngraph as ng
import numpy as np


class OpsReduction(OpsBase):
    """
    Mix-in class for reduction related ops
    """

    def _apply_reduction_op(self, reduction_op, tf_node, inputs):
        """
        Apply ngraph reduction op

        Args:
            reduction_op: ngraph op
            tf_node: tensorflow node
            inputs: list of ngraph op inputs

        Returns:
            The resulting ngraph node
        """
        # get inputs
        input_tensor, reduction_indices = inputs

        # check reduction_indices comes from constant
        try:
            assert reduction_indices.const is not None
        except:
            raise NotImplementedError("[NON-NATIVE] reduction_indices be "
                                      "constants, cannot come from intermediate "
                                      "results")
        # get out
        # reduction_indices = set([int(ind) for ind in reduction_indices.const])
        # input_shape = input_tensor.axes.lengths
        # input_ndims = len(input_shape)
        # out_axes = ng.Axes([input_tensor.axes[ind]
        #                     for ind in range(input_ndims)
        #                     if ind not in reduction_indices])
        reduction_indices = [int(ind) for ind in reduction_indices.const]
        reduction_axes = ng.Axes([input_tensor.axes[ind]
                                  for ind in reduction_indices])

        # perform reduction operation
        # result_op = reduction_op(input_tensor, out_axes=out_axes)
        result_op = reduction_op(input_tensor, reduction_axes=reduction_axes)

        # broadcast results for safety
        new_out_axes = [ng.Axis(length=axis.length) for axis in result_op.axes]
        result_op = ng.AxesCastOp(result_op, new_out_axes, name=tf_node.name)

        return result_op

    def Sum(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Computes the sum of elements across dimensions of a tensor.

        Reduces `input_tensor` along the dimensions given in `reduction_indices`.
        Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
        entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
        are retained with length 1.

        If `reduction_indices` has no entries, all dimensions are reduced, and a
        tensor with a single element is returned.

        For example:

        ```python
        # 'x' is [[1, 1, 1]
        #         [1, 1, 1]]
        tf.reduce_sum(x) ==> 6
        tf.reduce_sum(x, 0) ==> [2, 2, 2]
        tf.reduce_sum(x, 1) ==> [3, 3]
        tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
        tf.reduce_sum(x, [0, 1]) ==> 6
        ```

        Args:
            input_tensor: The tensor to reduce. Should have numeric type.
            reduction_indices: The dimensions to reduce. If `None` (the default),
                               reduces all dimensions.
            keep_dims: If true, retains reduced dimensions with length 1.
            name: A name for the operation (optional).

        Returns:
            The reduced tensor.
        """
        return self._apply_reduction_op(ng.sum, tf_node, inputs)

    def Mean(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Computes the mean of elements across dimensions of a tensor.

        Reduces `input_tensor` along the dimensions given in `reduction_indices`.
        Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
        entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
        are retained with length 1.

        If `reduction_indices` has no entries, all dimensions are reduced, and a
        tensor with a single element is returned.

        For example:

        ```python
        # 'x' is [[1., 1.]
        #         [2., 2.]]
        tf.reduce_mean(x) ==> 1.5
        tf.reduce_mean(x, 0) ==> [1.5, 1.5]
        tf.reduce_mean(x, 1) ==> [1., 2.]
        ```

        Args:
            input_tensor: The tensor to reduce. Should have numeric type.
            reduction_indices: The dimensions to reduce. If `None` (the default),
                               reduces all dimensions.
            keep_dims: If true, retains reduced dimensions with length 1.
            name: A name for the operation (optional).

        Returns:
            The reduced tensor.
        """
        return self._apply_reduction_op(ng.mean, tf_node, inputs)

    def Prod(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Computes the product of elements across dimensions of a tensor.

        Reduces `input_tensor` along the dimensions given in `reduction_indices`.
        Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
        entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
        are retained with length 1.

        If `reduction_indices` has no entries, all dimensions are reduced, and a
        tensor with a single element is returned.

        Args:
            input_tensor: The tensor to reduce. Should have numeric type.
            reduction_indices: The dimensions to reduce. If `None` (the default),
                               reduces all dimensions.
            keep_dims: If true, retains reduced dimensions with length 1.
            name: A name for the operation (optional).

        Returns:
            The reduced tensor.
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
            out_axes = [ng.Axis(input_tensor.axes[i].length)
                        for i in range(len(input_tensor.axes))
                        if i not in reduction_indices]

            # broadcast for safety
            result_op = ng.Constant(np_result, axes=out_axes)
            return result_op
        except:
            raise NotImplementedError("reduce_prod currently not supported in "
                                      "ngraph, can only handle constant case.")
