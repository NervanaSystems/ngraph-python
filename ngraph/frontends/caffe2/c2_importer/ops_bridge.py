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

from ngraph.frontends.caffe2.c2_importer.ops_constant import OpsConstant
from ngraph.frontends.caffe2.c2_importer.ops_nn import OpsNN
from ngraph.frontends.caffe2.c2_importer.ops_binary import OpsBinary
from ngraph.frontends.caffe2.c2_importer.ops_unary import OpsUnary


class OpsBridge(OpsConstant, OpsNN, OpsBinary, OpsUnary):
    """
    Bridging op between Caffe2 / ngraph.

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
    """

    def __init__(self):
        self.init_assign_op_names = set()

    def __call__(self, c2_op, input_ops):
        """
        Call Op based on `c2_op.name`. Mix-in functions must have same name
        as the `c2_op.name`.

        Arguments:
            c2_op (OperatorDef): a Caffe2 node
            input_ops (List): list of ngraph op

        Returns:
            The resulting ngraph op
        """
        op_type = c2_op.type

        if hasattr(self, op_type):
            return getattr(self, op_type)(c2_op, input_ops)
        else:
            # print(c2_op.name, "ignored.")
            return None
