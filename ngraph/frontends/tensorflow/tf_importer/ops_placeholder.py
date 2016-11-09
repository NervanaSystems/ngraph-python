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

from ngraph.frontends.tensorflow.tf_importer.utils import tf_to_shape_axes
from ngraph.frontends.tensorflow.tf_importer.ops_base import OpsBase
import ngraph as ng


class OpsPlaceholder(OpsBase):
    """
    Mix-in class placeholder op
    """

    def Placeholder(self, tf_node, inputs):
        """
        Inserts a placeholder for a tensor.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            dtype, shape, name
        """
        axes = tf_to_shape_axes(tf_node)
        ng_op = ng.placeholder(axes).named(tf_node.name)
        return ng_op
