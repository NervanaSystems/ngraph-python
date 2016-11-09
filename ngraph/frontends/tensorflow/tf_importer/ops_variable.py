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


class OpsVariable(OpsBase):
    """
    Mix-in class for variable op
    """

    def Variable(self, tf_node, inputs):
        """
        Creates a trainable variable.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.
        """

        # get axes
        try:
            axes = tf_to_shape_axes(tf_node.attr['shape'])
        except:
            raise NotImplementedError('Shape must be know prior to execution')

        return ng.variable(axes).named(tf_node.name)

    def Assign(self, tf_node, inputs):
        """
        Assign `value` to `ref`.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            ref, value, validate_shape, use_locking, name
        """
        """
        TODO: currently cannot fully support the TensorFlow semantics.
        1. Assign in TF returns the assigned tensor, in ngraph, it returns
           None
        2. In TF, is the assigned tensor is not used, then it retain the
           original value
        """
        ref, value = inputs
        assert ref.axes.lengths == value.axes.lengths, "shape not the same"
        value = ng.cast_axes(value, ref.axes)

        if tf_node.name in self.init_assign_op_names:
            with ng.Op.saved_user_deps():
                return ng.assign(ref, value)
        else:
            return ng.assign(ref, value)

    def AssignAdd(self, tf_node, inputs):
        """
        Assign `ref` + `value` to `ref`.
        Update 'ref' by adding 'value' to it.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            ref, value, use_locking, name
        """
        ref, value = inputs
        assert ref.axes.lengths == value.axes.lengths, "shape not the same"
        value = ng.cast_axes(value, ref.axes)

        if tf_node.name in self.init_assign_op_names:
            with ng.Op.saved_user_deps():
                return ng.assign(ref, ref + value)
        else:
            return ng.assign(ref, ref + value)

    def NoOp(self, tf_node, inputs):
        """
        Does nothing. Only useful to implement doall by applying dependencies.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.
        """

        if tf_node.name == "init":
            # TODO remove hardcoded name by passing in names for op
            return ng.doall(all=inputs)
        else:
            raise NotImplementedError
