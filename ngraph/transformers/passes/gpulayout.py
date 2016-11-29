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

from ngraph.transformers.passes.passes import PeepholeGraphPass
from ngraph.util.generics import generic_method
from ngraph.op_graph.convolution import ConvolutionOp, bprop_conv, update_conv
from ngraph.op_graph.pooling import PoolingOp, BpropPoolOp
from ngraph.op_graph.op_graph import Op, ContiguousOp


def _is_strides_contiguous(td):
    contiguous_strides = [td.dtype.itemsize]
    for dim in reversed(td.shape[1:]):
        contiguous_strides.insert(0, contiguous_strides[0] * dim)
    return (tuple(contiguous_strides) == td.strides)


class GPUTensorLayout(PeepholeGraphPass):
    """TODO."""
    @generic_method(dispatch_base_type=Op)
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO
        """
        pass

    @visit.on_type(ConvolutionOp)
    def visit(self, op):
        inputs = op.args[0]
        filters = op.args[1]

        inputs_td = inputs.tensor_description()
        filters_td = filters.tensor_description()
        replace = False

        if not _is_strides_contiguous(inputs_td):
            inputs = ContiguousOp(inputs)
            replace = True

        if not _is_strides_contiguous(filters_td):
            filters = ContiguousOp(filters)
            replace = True

        if replace:
            new_op = ConvolutionOp(op.conv_params, inputs, filters, axes=op.axes)
            self.replace_op(op, new_op)

    @visit.on_type(bprop_conv)
    def visit(self, op):
        deltas = op.args[0]
        filters = op.args[1]

        deltas_td = deltas.tensor_description()
        filters_td = filters.tensor_description()
        replace = False

        if not _is_strides_contiguous(deltas_td):
            deltas = ContiguousOp(deltas)
            replace = True

        if not _is_strides_contiguous(filters_td):
            filters = ContiguousOp(filters)
            replace = True

        if replace:
            new_op = bprop_conv(deltas, op.inputs, filters, op.fprop, axes=op.axes)
            self.replace_op(op, new_op)

    @visit.on_type(update_conv)
    def visit(self, op):
        deltas = op.args[0]
        inputs = op.args[1]

        deltas_td = deltas.tensor_description()
        inputs_td = inputs.tensor_description()
        replace = False

        if not _is_strides_contiguous(deltas_td):
            deltas = ContiguousOp(deltas)
            replace = True

        if not _is_strides_contiguous(inputs_td):
            inputs = ContiguousOp(inputs)
            replace = True

        if replace:
            new_op = update_conv(deltas, inputs, op.filters, op.fprop, axes=op.axes)
            self.replace_op(op, new_op)

    @visit.on_type(PoolingOp)
    def visit(self, op):
        inputs = op.args[0]
        inputs_td = inputs.tensor_description()

        if not _is_strides_contiguous(inputs_td):
            inputs = ContiguousOp(inputs)
            new_op = PoolingOp(op.pool_params, inputs, axes=op.axes)
            self.replace_op(op, new_op)

    @visit.on_type(BpropPoolOp)
    def visit(self, op):
        deltas = op.args[0]
        deltas_td = deltas.tensor_description()

        if not _is_strides_contiguous(deltas_td):
            deltas = ContiguousOp(deltas)
            new_op = BpropPoolOp(deltas, op.inputs, op.fprop, axes=op.axes)
            self.replace_op(op, new_op)
