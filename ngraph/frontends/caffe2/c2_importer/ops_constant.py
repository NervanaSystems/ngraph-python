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

from ngraph.frontends.caffe2.c2_importer.ops_base import OpsBase
# TODO: temporary dependency on tf_importer
from ngraph.frontends.tensorflow.tf_importer.utils import shape_to_axes
import ngraph as ng
import numpy as np

class OpsConstant(OpsBase):
    """
    Mix-in class for constant ops.
    """

    def ConstantFill(self, c2_node, inputs):
        """
        Creates a constant tensor.

        Arguments:
            c2_node: NodeDef object, the caffe2 node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the caffe2 node.

        Inputs to c2_node:
            value, dtype, shape, name
        """
        # convert to numpy value

        shape =  None
        value = None
        # TBD: There has to be better way to do this
        for arg in c2_node.arg:
            if arg.name == "shape":
                shape = arg.ints
            if arg.name == "value":
                value = arg.f
        np_val = np.full(tuple(shape), value)

        ng_op = ng.constant(np_val,
                            shape_to_axes(np_val.shape)).named(c2_node.name)
        return ng_op