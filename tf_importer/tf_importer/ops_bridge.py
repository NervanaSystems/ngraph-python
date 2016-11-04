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

from tf_importer.tf_importer.ops_binary import OpsBinary
from tf_importer.tf_importer.ops_constant import OpsConstant
from tf_importer.tf_importer.ops_placeholder import OpsPlaceholder
from tf_importer.tf_importer.ops_unary import OpsUnary
from tf_importer.tf_importer.ops_matmul import OpsMatmul
from tf_importer.tf_importer.ops_reduction import OpsReduction
from tf_importer.tf_importer.ops_variable import OpsVariable
from tf_importer.tf_importer.ops_transform import OpsTransform
from tf_importer.tf_importer.ops_nn import OpsNN
from tf_importer.tf_importer.ops_gradient import OpsGradient


class OpsBridge(OpsConstant,
                OpsBinary,
                OpsPlaceholder,
                OpsUnary,
                OpsMatmul,
                OpsReduction,
                OpsVariable,
                OpsTransform,
                OpsNN,
                OpsGradient):
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

    def __init__(self):
        self.init_assign_op_names = set()

    def __call__(self, tf_node, input_ops):
        """
        Call Op based on `tf_node.name`. Mix-in functions must have same name
        as the `tf_node.name`.

        Args:
            tf_node (NodeDef): a TensorFlow node
            input_ops (List): list of ngraph op

        Returns:
            The resulting ngraph op
        """
        op_name = tf_node.op

        if hasattr(self, op_name):
            return getattr(self, op_name)(tf_node, input_ops)
        else:
            return None
