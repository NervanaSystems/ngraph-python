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
from __future__ import division

import ngraph as ng


class OpsCompound:
    """
    Bridging compoud operations between CNTK and ngraph.
    """

    def Dense(self, cntk_op, inputs):
        """
        Computes fully-connected layer with optional activation function.

        Arguments:
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        return_op = cntk_op.block_root.uid

        block_ops = []
        stack = [cntk_op.block_root]
        while stack:
            node = stack.pop()
            node = node.root_function

            if node in block_ops:
                continue
            else:
                block_ops.append(node)

            for i in node.inputs:
                if i.is_output:
                    stack.append(i.owner)

        imported_ops = dict()
        while block_ops:
            node = block_ops.pop()
            node_inputs = []
            for i in node.inputs:
                if i.is_placeholder:
                    temp = next(iter([
                        v for v in inputs if not isinstance(v, ng.AssignableTensorOp)
                    ]))
                elif i.is_output:
                    temp = imported_ops.get(i.owner.root_function.uid)
                else:
                    temp = next(iter([
                        v for v in inputs if v.name == i.uid
                    ]))

                if temp is not None:
                    node_inputs.append(temp)
                else:
                    raise ValueError("Unknown input: " + i.uid)
            try:
                imported_ops[node.uid] = getattr(self, node.op_name)(node, node_inputs)
            except AttributeError:
                raise TypeError("Unknown operation: " + node.op_name)

        return imported_ops[return_op]

    def CrossEntropyWithSoftmax(self, cntk_op, inputs):
        """
        Computes the softmax cross entropy between the inputs[0] and inputs[1].

        Arguments:
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        cast_0, cast_1 = inputs

        if cast_0.axes.lengths != cast_1.axes.lengths:
            cast_0 = ng.Transpose(cast_0)
        assert cast_0.axes.lengths == cast_1.axes.lengths

        cast_0 = ng.cast_axes(cast_0, axes=cast_1.axes)
        loss = ng.cross_entropy_multi(ng.softmax(cast_0), cast_1)

        return ng.mean(loss, out_axes=()).named(cntk_op.uid)

    def Combine(self, cntk_op, inputs):
        """
        Returns combined outputs of inputs list.

        Arguments:
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        return ng.stack(inputs, ng.make_axis(len(inputs))).named(cntk_op.uid)
