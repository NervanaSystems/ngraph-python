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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tf_importer.tf_importer.utils import shape_to_axes
from tf_importer.tf_importer.ops_base import OpsBase
from tensorflow.python.framework import tensor_util
import ngraph as ng
import numpy as np
import scipy.stats


class OpsConstant(OpsBase):
    """
    Mix-in class for unary ops.
    """

    def Const(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Creates a constant tensor.

        The resulting tensor is populated with values of type `dtype`, as
        specified by arguments `value` and (optionally) `shape` (see examples
        below).

        The argument `value` can be a constant value, or a list of values of type
        `dtype`. If `value` is a list, then the length of the list must be less
        than or equal to the number of elements implied by the `shape` argument (if
        specified). In the case where the list length is less than the number of
        elements specified by `shape`, the last element in the list will be used
        to fill the remaining entries.

        The argument `shape` is optional. If present, it specifies the dimensions of
        the resulting tensor. If not present, the shape of `value` is used.

        If the argument `dtype` is not specified, then the type is inferred from
        the type of `value`.

        For example:

        ```python
        # Constant 1-D Tensor populated with value list.
        tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]

        # Constant 2-D tensor populated with scalar value -1.
        tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]
                                                    [-1. -1. -1.]]
        ```

        Args:
            value:  A constant value (or list) of output type `dtype`.

            dtype:  The type of the elements of the resulting tensor.

            shape:  Optional dimensions of resulting tensor.

            name:   Optional name for the tensor.

        Returns:
            A Constant Tensor.

        TensorFlow provides several operations that you can use to generate
        constants. More specifically,

            tf.zeros(shape, dtype=tf.float32, name=None)
            tf.zeros_like(tensor, dtype=None, name=None)
            tf.ones(shape, dtype=tf.float32, name=None)
            tf.ones_like(tensor, dtype=None, name=None)
            tf.fill(dims, value, name=None)
            tf.constant(value, dtype=None, shape=None, name=Const)

        They all create op: "Const".
        """
        # convert to numpy value
        np_val = tensor_util.MakeNdarray(tf_node.attr['value'].tensor)
        ng_op = ng.Constant(np_val, axes=shape_to_axes(np_val.shape),
                            name=tf_node.name)
        return ng_op

    def Fill(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Creates a tensor filled with a scalar value.

        This operation creates a tensor of shape `dims` and fills it with `value`.

        For example:

        ```prettyprint
        # Output tensor has shape [2, 3].
        fill([2, 3], 9) ==> [[9, 9, 9]
                           [9, 9, 9]]
        ```

        Args:
            dims: A `Tensor` of type `int32`.
                  1-D. Represents the shape of the output tensor.
            value: A `Tensor`. 0-D (scalar). Value to fill the returned tensor.
            name: A name for the operation (optional).

        Returns:
            A `Tensor`. Has the same type as `value`.
        """
        # get inputs
        shape_op, const_val_op = inputs

        # get shape, const_val
        shape = tuple(shape_op.const.astype(int))
        const_val = const_val_op.const

        # convert to numpy value
        np_val = np.zeros(shape)
        np_val.fill(const_val)

        # create op
        ng_op = ng.Constant(np_val, axes=shape_to_axes(np_val.shape),
                            name=tf_node.name)
        return ng_op

    def TruncatedNormal(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Outputs random values from a truncated normal distribution.

        The generated values follow a normal distribution with specified mean and
        standard deviation, except that values whose magnitude is more than 2 standard
        deviations from the mean are dropped and re-picked.

        Args:
            shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
            mean: A 0-D Tensor or Python value of type `dtype`. The mean of the
                  truncated normal distribution.
                  stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
                  of the truncated normal distribution.
            dtype: The type of the output.
            seed: A Python integer. Used to create a random seed for the distribution.
                  See
                  [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
                  for behavior.
            name: A name for the operation (optional).

        Returns:
            A tensor of the specified shape filled with random truncated normal values.


        tf.truncated_normal() call generates several ops

        shape --> TruncatedNormal
                       |
                       V
        stddev -----> Mul
                       |
                       V
        mean -------> Add
                       |
                       V
                    (output)
        """
        # get inputs
        shape = tuple(inputs[0].const.astype(int))

        # generate truncated standard normal
        mu, sigma, lo, up = 0., 1., -2., 2
        generator = scipy.stats.truncnorm((lo - mu) / sigma, (up - mu) / sigma,
                                          loc=mu, scale=sigma)
        np_val = generator.rvs(shape)
        ng_op = ng.Constant(np_val, axes=shape_to_axes(np_val.shape),
                            name=tf_node.name)
        return ng_op

    def RandomStandardNormal(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Outputs random values from a normal distribution.

        Args:
            shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
            mean: A 0-D Tensor or Python value of type `dtype`. The mean of the normal
                  distribution.
                  stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
                  of the normal distribution.
            dtype: The type of the output.
            seed: A Python integer. Used to create a random seed for the distribution.
                  See
                  [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
                  for behavior.
            name: A name for the operation (optional).

        Returns:
            A tensor of the specified shape filled with random normal values.

        Similar to tf.truncated_normal(), tf.random_normal() call generates
        several ops. The `RandomStandardNormal` op is what we implement here.
        """
        # get inputs
        shape = tuple(inputs[0].const.astype(int))

        # generate standard normal
        np_val = np.random.standard_normal(size=shape)
        ng_op = ng.Constant(np_val, axes=shape_to_axes(np_val.shape),
                            name=tf_node.name)
        return ng_op

    def ZerosLike(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Creates a tensor with all elements set to zero.

        Given a single tensor (`tensor`), this operation returns a tensor of the
        same type and shape as `tensor` with all elements set to zero. Optionally,
        you can use `dtype` to specify a new type for the returned tensor.

        For example:

        ```python
        # 'tensor' is [[1, 2, 3], [4, 5, 6]]
        tf.zeros_like(tensor) ==> [[0, 0, 0], [0, 0, 0]]
        ```

        Args:
            tensor: A `Tensor`.
            dtype: A type for the returned `Tensor`. Must be `float32`, `float64`,
                   `int8`, `int16`, `int32`, `int64`, `uint8`, `complex64`, or `complex128`.
            name: A name for the operation (optional).

        Returns:
            A `Tensor` with all elements set to zero.
        """
        shape = inputs[0].axes.lengths
        np_val = np.zeros(shape)
        ng_op = ng.Constant(np_val, axes=shape_to_axes(np_val.shape),
                            name=tf_node.name)
        return ng_op
