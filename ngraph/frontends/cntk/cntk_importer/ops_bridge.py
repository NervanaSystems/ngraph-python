# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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

from ngraph.frontends.cntk.cntk_importer.ops_compound import OpsCompound
from ngraph.frontends.cntk.cntk_importer.ops_binary import OpsBinary
from ngraph.frontends.cntk.cntk_importer.ops_unary import OpsUnary


class OpsBridge(OpsCompound, OpsBinary, OpsUnary):
    """
    Bridging operations between CNTK and ngraph.
    """

    def __call__(self, cntk_op, inputs):
        """
        Call Op based on `cntk_op.op_name`.

        Arguments:
            cntk_op: CNTK operation to be translated.
            inputs: List of prepared inputs for operation.

        Returns:
            The resulting ngraph op.
        """
        op_name = cntk_op.op_name
        try:
            return getattr(self, op_name)(cntk_op, inputs)
        except AttributeError:
            raise TypeError("Unknown operation: " + op_name)
