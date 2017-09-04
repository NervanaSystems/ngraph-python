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

import ngraph as ng


class OpsUnary:
    """
    Bridging unary operations between CNTK and ngraph.
    """

    def Negate(self, cntk_op, inputs):
        """
        Returns element-wise negation of inputs[0].

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 1

        return ng.negative(inputs[0]).named(cntk_op.uid)

    def Log(self, cntk_op, inputs):
        """
        Returns element-wise natural logarithm of inputs[0].

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 1

        return ng.LogOp(inputs[0]).named(cntk_op.uid)

    def Sqrt(self, cntk_op, inputs):
        """
        Returns element-wise square-root of inputs[0].

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 1

        return ng.sqrt(inputs[0]).named(cntk_op.uid)

    def Floor(self, cntk_op, inputs):
        """
        Returns element-wise value rounded to the largest integer less than or equal to inputs[0].

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 1

        return ng.subtract(inputs[0], ng.mod(inputs[0], 1)).named(cntk_op.uid)

    def Sigmoid(self, cntk_op, inputs):
        """
        Returns element-wise sigmoid of inputs[0].

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 1

        return ng.sigmoid(inputs[0]).named(cntk_op.uid)

    def StableSigmoid(self, cntk_op, inputs):
        """
        Returns element-wise sigmoid of inputs[0].

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 1

        return ng.sigmoid(inputs[0]).named(cntk_op.uid)

    def Exp(self, cntk_op, inputs):
        """
        Returns element-wise exp of inputs[0].

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 1

        return ng.exp(inputs[0]).named(cntk_op.uid)

    def Tanh(self, cntk_op, inputs):
        """
        Returns element-wise tanh of inputs[0].

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 1

        return ng.tanh(inputs[0]).named(cntk_op.uid)

    def Abs(self, cntk_op, inputs):
        """
        Returns element-wise absolute of inputs[0].

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 1

        return ng.absolute(inputs[0]).named(cntk_op.uid)

    def Reciprocal(self, cntk_op, inputs):
        """
        Returns element-wise reciprocal of inputs[0].

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 1

        return ng.reciprocal(inputs[0]).named(cntk_op.uid)

    def ReduceElements(self, cntk_op, inputs):
        """
        Returns a reduction operation (max, min, mean, sum, prod) or a calculation which matches
        CNTK's LogSum reduction (`reduce_log_sum_exp` function).

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 1

        reduction_op_name = cntk_op.attributes.get('reductionOpName')
        # CNTK API defines a reductionKeepDimensions flag, but we currently don't use it
        # keep_dimensions = cntk_op.attributes.get('reductionKeepDimensions', False)

        cntk_op_attribute_axes = []
        if cntk_op.attributes.get('axisVec'):
            cntk_op_attribute_axes.extend(cntk_op.attributes.get('axisVec'))
        elif cntk_op.attributes.get('axis'):
            cntk_op_attribute_axes.append(cntk_op.attributes.get('axis'))

        # CNTK axes are numbered in reverse order: the last axis is labeled 0, the previous 1, etc.
        reduction_axes_indexes = [len(inputs[0].axes) - 1 - i
                                  for (_, _, i) in cntk_op_attribute_axes]
        reduction_ng_axes_list = [axis for (i, axis) in enumerate(inputs[0].axes)
                                  if i in reduction_axes_indexes]
        reduction_ng_axes = ng.Axes(axes=reduction_ng_axes_list)

        if reduction_op_name == 'Max':
            return ng.max(inputs[0], reduction_axes=reduction_ng_axes).named(cntk_op.uid)

        if reduction_op_name == 'Min':
            return ng.min(inputs[0], reduction_axes=reduction_ng_axes).named(cntk_op.uid)

        if reduction_op_name == 'Mean':
            return ng.mean(inputs[0], reduction_axes=reduction_ng_axes).named(cntk_op.uid)

        if reduction_op_name == 'Sum':
            return ng.sum(inputs[0], reduction_axes=reduction_ng_axes).named(cntk_op.uid)

        if reduction_op_name == 'Prod':
            return ng.prod(inputs[0], reduction_axes=reduction_ng_axes).named(cntk_op.uid)

        if reduction_op_name == 'LogSum':
            return ng.log(ng.sum(ng.exp(inputs[0]), reduction_axes=reduction_ng_axes))\
                .named(cntk_op.uid)

        raise NotImplementedError('CNTKImporter: ReduceElements does not support operation %s',
                                  reduction_op_name)

    def ReLU(self, cntk_op, inputs):
        """
        Returns element-wise rectified linear of inputs[0].

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 1

        return ng.maximum(inputs[0], 0.).named(cntk_op.uid)

    def Reshape(self, cntk_op, inputs):
        """
        Returns input having reinterpreted tensor dimensions.

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        assert len(inputs) == 1

        in_axes = list(inputs[0].axes)
        out_axes = []
        for dim in cntk_op.shape:
            found = False
            for axis in in_axes:
                if axis.length == dim:
                    found = True
                    out_axes.append(axis)
                    in_axes.remove(axis)
                    break
            if found is not True:
                out_axes.append(ng.make_axis(dim))

        out_axes += in_axes
        return ng.broadcast(inputs[0], out_axes).named(cntk_op.uid)
