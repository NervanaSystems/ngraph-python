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


class OpsGradient(OpsBase):
    """
    Mix-in class for gradient related ops
    """

    def ReluGrad(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        C++ implementation: https://goo.gl/l07FXx

        Computes ReluGrad backprops.

        NOTE: When the activation is exactly zero, we do not propagate the
        associated gradient value. This allows the output of the Relu to be used,
        as well as its input.

        Args:
            gradients: gradients backpropagated to the Relu op.
            features: either the inputs that were passed to the Relu or, or its
                      outputs (using either one yields the same result here).


        Returns:
            backprops: gradients to backpropagate to the Relu inputs.
        """
        # get inputs
        gradients, features = inputs

        # gradient of relu op
        relu_grad = ng.greater(features, 0.)
        relu_grad = ng.AxesCastOp(relu_grad, gradients.axes)

        return gradients * relu_grad
