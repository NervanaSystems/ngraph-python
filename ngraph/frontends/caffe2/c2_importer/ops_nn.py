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
from ngraph.frontends.neon import ar
from ngraph.op_graph.axes import spatial_axis
import ngraph as ng
import numpy as np


class OpsNN(OpsBase):
    """
    Mix-in class for NN ops:
    """
    def FC(self, c2_op, inputs):
        """
        Multiplies matrix `a` by matrix `b`. The inputs must be two-dimensional,
        the inner dimensions must match (possibly after transpose).

        Arguments:
            c2_op: OperatorDef object, the caffe2 node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the caffe2 node.

        Inputs to c2_op:
            a, b, transpose_a, transpose_b, a_is_sparse, b_is_sparse, name
        """
        # get inputs
        left, right, bias = inputs

        # check shape
        assert left.axes[1].length == right.axes[1].length

        # cast axis
        left_casted = ng.cast_axes(left, [left.axes[0], right.axes[1] - 1])

        # add op
        add_op = ng.dot(left_casted, right)

        # cast bias axis
        bias_casted = ng.cast_axes(bias, [add_op.axes[-1]])

        # result op
        result_op = ng.add(add_op, bias_casted, name=c2_op.name)
        return result_op

    def AveragedLoss(self, c2_op, inputs):
        """
        Computes average loss for the batch.

        Arguments:
            c2_op: OperatorDef object, the caffe2 node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the caffe2 node.
        """
        return ng.mean(inputs[0], reduction_axes=inputs[0].axes.batch_axes())

    def LabelCrossEntropy(self, c2_op, inputs):
        """
        Computes the cross entropy between the input and the label set.

        Arguments:
            c2_op: OperatorDef object, the caffe2 node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the caffe2 node.
       """
        y, labels = inputs

        labels_one_hot = ng.one_hot(labels, axis=y.axes[1])
        labels_one_hot = ng.cast_axes(labels_one_hot, [labels_one_hot.axes[0], y.axes[0]])
        return ng.cross_entropy_multi(y, labels_one_hot, out_axes=y.axes[0])

    def MaxPool(self, c2_op, inputs):
        return self.Pool(c2_op, inputs)

    def AveragePool(self, c2_op, inputs):
        return self.Pool(c2_op, inputs)

    def Pool(self, c2_op, inputs):
        """
        Performs max or average pooling on the input.

        Arguments:
            c2_op: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the c2_op node.

        Inputs to c2_op:
            input
        """
        supported_pooling = {'MaxPool': 'max', 'AveragePool': 'avg'}

        image = inputs[0]

        # TODO: we assume NCHW, make some assert here?

        # set axes shape
        ax_N = ng.make_axis(batch=True)
        ax_C = ng.make_axis(roles=[ar.Channel])
        ax_D = ng.make_axis(roles=[ar.Depth])
        ax_H = ng.make_axis(roles=[ar.Height])
        ax_W = ng.make_axis(roles=[ar.Width])
        ng.make_axes([ax_N, ax_C, ax_H, ax_W]).set_shape(image.axes.lengths)
        ax_D.length = 1

        # spatial kernel size
        kernel_size = [int(val.i) for val in c2_op.arg._values if val.name == "kernel"]
        if len(kernel_size) != 1:
            raise ValueError("Kernel size must be scalar value")
        # kernel is square
        kernel_h = kernel_w = kernel_size[0]
        kernel_d = kernel_c = 1

        # strides params
        stride_size = [int(val.i) for val in c2_op.arg._values if val.name == "stride"]
        if len(kernel_size) != 1:
            raise ValueError("Stride size must be scalar value")
        stride_h = stride_w = stride_size[0]

        # padding params

        # TODO: how to handle padding in caffe2?
        # padding = c2_op.attr['padding'].s.decode("ascii")
        # padding = (image_size - kernel_size) % stride_size
        padding = np.mod(np.array(image.axes.lengths) - np.array([1, 1, kernel_h, kernel_w]),
                         np.array([1, 1, stride_size[0], stride_size[0]]))
        if not np.array_equal(padding, [0] * len(padding)):
            raise NotImplementedError("Max pooling does not support padding yet")

        pad_t = pad_b = pad_l = pad_r = 0

        if pad_t != pad_b or pad_l != pad_r:
            raise NotImplementedError("Requires symmetric padding in ngraph:"
                                      "pad_t(%s) == pad_b(%s) and"
                                      "pad_l(%s) == pad_r(%s)" %
                                      (pad_t, pad_b, pad_l, pad_r))

        # pooling params
        params = dict(op=supported_pooling[c2_op.type],
                      pad_d=0, pad_h=pad_t, pad_w=pad_l, pad_c=0,
                      str_d=1, str_h=stride_h, str_w=stride_w, str_c=1,
                      J=kernel_c, T=kernel_d, R=kernel_h, S=kernel_w)

        # i, f, o axes
        ax_i = ng.make_axes([ax_C, ax_D, ax_H, ax_W, ax_N])
        ax_o = ng.make_axes([
            spatial_axis(ax_i, kernel_c, params['pad_c'], params['str_c'], ar.Channel),
            spatial_axis(ax_i, kernel_d, params['pad_d'], params['str_d'], ar.Depth),
            spatial_axis(ax_i, kernel_h, params['pad_h'], params['str_h'], ar.Height),
            spatial_axis(ax_i, kernel_w, params['pad_w'], params['str_w'], ar.Width),
            ax_N
        ])

        # broadcast input / filter axes
        image = ng.cast_axes(image, ng.make_axes([ax_N, ax_C, ax_H, ax_W]))
        image = ng.expand_dims(image, ax_D, 1)  # NCHW -> NDCHW
        image = ng.axes_with_order(image, axes=ax_i)  # NDCHW -> CDHWN

        # pooling
        output = ng.pooling(params, image, axes=ax_o)

        # cast back to NDCHW
        oC, oD, oH, oW, oN = output.axes
        output = ng.broadcast(output, ng.make_axes([oN, oD, oC, oH, oW]))

        # slice away the oD
        out_slicing = [slice(None), 0, slice(None), slice(None), slice(None)]
        output = ng.tensor_slice(output, out_slicing)

        return output
