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


class OpsMatmul(OpsBase):
    """
    Mix-in class for ops:
        - Matmul
    """

    def MatMul(self, tf_node, inputs):
        """
        TODO: support transpose

        [TensorFlow Docs]
        Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

        The inputs must be two-dimensional matrices, with matching inner dimensions,
        possibly after transposition.

        Both matrices must be of the same type. The supported types are:
        `float`, `double`, `int32`, `complex64`.

        Either matrix can be transposed on the fly by setting the corresponding flag
        to `True`. This is `False` by default.

        If one or both of the matrices contain a lot of zeros, a more efficient
        multiplication algorithm can be used by setting the corresponding
        `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.

        For example:

        ```python
        # 2-D tensor `a`
        a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) => [[1. 2. 3.]
                                                            [4. 5. 6.]]
        # 2-D tensor `b`
        b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2]) => [[7. 8.]
                                                               [9. 10.]
                                                               [11. 12.]]
        c = tf.matmul(a, b) => [[58 64]
                              [139 154]]
        ```

        Args:
            a: `Tensor` of type `float`, `double`, `int32` or `complex64`.
            b: `Tensor` with same type as `a`.
            transpose_a: If `True`, `a` is transposed before multiplication.
                         transpose_b: If `True`, `b` is transposed before multiplication.
                         a_is_sparse: If `True`, `a` is treated as a sparse matrix.
                         b_is_sparse: If `True`, `b` is treated as a sparse matrix.
            name: Name for the operation (optional).

        Returns:
            A `Tensor` of the same type as `a`.
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

        # cast axis
        right_axes = ng.make_axes([left.axes[1], right.axes[1]])
        right_casted = ng.cast_axes(right, axes=right_axes)

        # result op
        result_op = ng.dot(left, right_casted, name=tf_node.name)

        # return
        return result_op
