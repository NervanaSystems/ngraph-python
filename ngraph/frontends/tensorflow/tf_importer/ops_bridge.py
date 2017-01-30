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

from __future__ import print_function
from ngraph.frontends.tensorflow.tf_importer.ops_variable import OpsVariable
from ngraph.frontends.tensorflow.tf_importer.ops_binary import OpsBinary
from ngraph.frontends.tensorflow.tf_importer.ops_constant import OpsConstant
from ngraph.frontends.tensorflow.tf_importer.ops_gradient import OpsGradient
from ngraph.frontends.tensorflow.tf_importer.ops_matmul import OpsMatmul
from ngraph.frontends.tensorflow.tf_importer.ops_nn import OpsNN
from ngraph.frontends.tensorflow.tf_importer.ops_placeholder import OpsPlaceholder
from ngraph.frontends.tensorflow.tf_importer.ops_reduction import OpsReduction
from ngraph.frontends.tensorflow.tf_importer.ops_transform import OpsTransform
from ngraph.frontends.tensorflow.tf_importer.ops_unary import OpsUnary


class OpsBridge(OpsConstant, OpsBinary, OpsPlaceholder, OpsUnary, OpsMatmul,
                OpsReduction, OpsVariable, OpsTransform, OpsNN, OpsGradient):
    """
    Bridging op between TensorFlow / ngraph.

    OpsBase
        ^
        |_____________________________________________________ ...
        |                 |                 |
    OpsBinary         OpsUnary           OpsReduction          ...
        ^                 ^                 ^
        |def Add()        |def Tanh()       |
        |def Mul()        |def Sigmoid()    |
        |...              |...              |
        |_________________|_________________|_________________ ...
        |
        |
    OpsBridge (contains mix-ins from OpsBinary, OpsUnary, ...)

    TODO: Organize ops as in TensorFlow's directory structure
    """
    def __call__(self, tf_node, input_ops):
        """
        Call Op based on `tf_node.name`. Mix-in functions must have same name
        as the `tf_node.name`.

        Arguments:
            tf_node (NodeDef): a TensorFlow node
            input_ops (List): list of ngraph op

        Returns:
            The resulting ngraph op
        """
        op_name = tf_node.op

        # if op not handled, gets -1
        ng_op = getattr(self, op_name, None)

        if ng_op:
            return ng_op(tf_node, input_ops)
        else:
            # ignored op set to None
            print(tf_node.name, "ignored.")
            return None
