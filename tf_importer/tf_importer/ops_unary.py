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


class OpsUnary(OpsBase):
    """
    Mix-in class for unary ops
    """

    def _element_wise_unary(self, ng_op, tf_node, inputs):
        # get inputs
        left = inputs[0]

        # result
        result_op = ng_op(left, name=tf_node.name)

        # return op
        return result_op

    def Tanh(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Computes hyperbolic tangent of `x` element-wise.

        Args:
            x: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
               or `qint32`.
            name: A name for the operation (optional).

        Returns:
            A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
                the return type is `quint8`.
        """
        return self._element_wise_unary(ng.tanh, tf_node, inputs)

    def Sigmoid(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Computes sigmoid of `x` element-wise.

        Specifically, `y = 1 / (1 + exp(-x))`.

        Args:
            x: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
                or `qint32`.
            name: A name for the operation (optional).

        Returns:
            A Tensor with the same type as `x` if `x.dtype != qint32`
                otherwise the return type is `quint8`.
        """
        return self._element_wise_unary(ng.sigmoid, tf_node, inputs)

    def Relu(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Computes rectified linear: `max(features, 0)`.

        Args:
            features: A `Tensor`. Must be one of the following types: `float32`,
                      `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`,
                      `uint16`, `half`.
            name: A name for the operation (optional).

        Returns:
            A `Tensor`. Has the same type as `features`.
        """
        return ng.maximum(inputs[0], 0., name=tf_node.name)

    def Identity(self, tf_node, inputs):
        """
        TODO: implement as a copy operation

        [TensorFlow Docs]
        Return a tensor with the same shape and contents as the input tensor or value.

        Args:
            input: A `Tensor`.
            name: A name for the operation (optional).

        Returns:
            A `Tensor`. Has the same type as `input`.

        A control flow operation used for enforcing dependencies.
        """
        return inputs[0]

    def Log(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Computes natural logarithm of x element-wise.

        I.e., \\(y = \log_e x\\).

        Args:
            x: A `Tensor`. Must be one of the following types: `half`,
               `float32`, `float64`, `complex64`, `complex128`.
            name: A name for the operation (optional).

        Returns:
            A `Tensor`. Has the same type as `x`.
        """
        return ng.log(inputs[0], name=tf_node.name)

    def Neg(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Computes numerical negative value element-wise.

        I.e., \\(y = -x\\).

        Args:
            x: A `Tensor`. Must be one of the following types: `half`,
               `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
            name: A name for the operation (optional).

        Returns:
            A `Tensor`. Has the same type as `x`.
        """
        return ng.negative(inputs[0], name=tf_node.name)
