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
from ngraph.frontends.neon.layer import output_dim
from ngraph.frontends.common.utils import common_conv2d_pool_padding
import ngraph as ng
import numpy as np


def _c2_padding(c2_op, in_NHWC, kernel_HWIO, stride_NHWC):
    pad_num = [val.i for val in c2_op.arg._values if val.name == 'legacy_pad']
    pad_dict = {0: 'NOTSET', 1: 'VALID', 2: 'SAME', 3: 'CAFFE_LEGACY_POOLING'}
    if not pad_num:
        pad_num = 0
    elif len(pad_num) != 1:
        raise ValueError(c2_op.type + " uses multiple paddings")
    else:
        pad_num = pad_num[0]

    if pad_num >= 0 and pad_num <= 3:
        pad_type = pad_dict[pad_num]
    else:
        raise ValueError(c2_op.type + " uses unknown padding")

    if 'NOTSET' == pad_type:
        padding = np.mod(np.array(in_NHWC) - np.array([1] + kernel_HWIO[:-1]),
                         np.array(stride_NHWC))
        if not np.array_equal(padding, [0] * len(padding)):
            raise NotImplementedError(c2_op.type + " padding type is not defined.")
    else:
        padding = common_conv2d_pool_padding(
            in_NHWC=in_NHWC,
            f_HWIO=kernel_HWIO,
            str_NHWC=stride_NHWC,
            padding=pad_type)
    return padding


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
        dot_op = ng.dot(left_casted, right)
        # cast bias axis
        bias_casted = ng.cast_axes(bias, [dot_op.axes[-1]])
        # result op
        result_op = ng.add(dot_op, bias_casted)
        return result_op

    def SquaredL2Distance(self, c2_op, inputs):
        """
        Computes squared L2 distance between two inputs.

        Arguments:
            c2_op: OperatorDef object, the caffe2 node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the caffe2 node.
        """
        x, y = inputs

        y = ng.cast_axes(y, x.axes)
        return 0.5 * ng.squared_L2(x - y)

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

        # set input axes shape
        ax_N = ng.make_axis(batch=True)
        ax_C = ng.make_axis(roles=[ar.Channel])
        ax_D = ng.make_axis(roles=[ar.Depth], length=1)
        ax_H = ng.make_axis(roles=[ar.Height])
        ax_W = ng.make_axis(roles=[ar.Width])
        ng.make_axes([ax_N, ax_C, ax_H, ax_W]).set_shape(image.axes.lengths)

        # create placeholders for output axes
        oC = ng.make_axis(roles=[ar.Channel]).named('C')
        oD = ng.make_axis(roles=[ar.Depth], length=1).named('D')
        oH = ng.make_axis(roles=[ar.Height]).named('H')
        oW = ng.make_axis(roles=[ar.Width]).named('W')

        # spatial kernel size
        kernel_size = [int(val.i) for val in c2_op.arg._values if val.name == "kernel"]
        if len(kernel_size) != 1:
            raise ValueError("Kernel size must be scalar value")
        # kernel is square
        kernel_h = kernel_w = kernel_size[0]
        kernel_d = kernel_c = 1

        # strides params
        stride_size = [int(val.i) for val in c2_op.arg._values if val.name == "stride"]
        if len(stride_size) != 1:
            raise ValueError("Stride size must be scalar value")
        stride_h = stride_w = stride_size[0]

        # padding params
        pad_t, pad_b, pad_l, pad_r = \
            _c2_padding(c2_op,
                        in_NHWC=[ax_N.length, ax_H.length, ax_W.length, ax_C.length],
                        kernel_HWIO=[kernel_h, kernel_w, ax_C.length, ax_C.length],
                        stride_NHWC=[1, stride_h, stride_w, 1])
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

        # i, o axes
        oC.length = output_dim(ax_C.length, kernel_c, params['pad_c'], params['str_c'])
        oD.length = output_dim(ax_D.length, kernel_d, params['pad_d'], params['str_d'])
        oH.length = output_dim(ax_H.length, kernel_h, params['pad_h'], params['str_h'])
        oW.length = output_dim(ax_W.length, kernel_w, params['pad_w'], params['str_w'])
        ax_i = ng.make_axes([ax_C, ax_D, ax_H, ax_W, ax_N])
        ax_o = ng.make_axes([oC, oD, oH, oW, ax_N])

        # broadcast input / filter axes
        image = ng.cast_axes(image, ng.make_axes([ax_N, ax_C, ax_H, ax_W]))
        image = ng.expand_dims(image, ax_D, 1)  # NCHW -> NDCHW
        image = ng.axes_with_order(image, axes=ax_i)  # NDCHW -> CDHWN

        # pooling
        output = ng.pooling(params, image, axes=ax_o)

        # cast back to NDCHW
        output = ng.broadcast(output, ng.make_axes([ax_N, oD, oC, oH, oW]))

        # slice away the oD
        out_slicing = [slice(None), 0, slice(None), slice(None), slice(None)]
        output = ng.tensor_slice(output, out_slicing)

        return output

    def StopGradient(self, c2_op, inputs):
        """ TODO """
        assert 1 == len(inputs)
        return ng.stop_gradient(inputs[0])

    def Conv(self, c2_op, inputs):
        """
        Computes a 2-D convolution given 4D input and filter tensors.

        Arguments:
            c2_op: NodeDef object, the caffe2 node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the caffe2 node.

        Inputs to c2_op:
            input, wegiths, filter

        Supports caffe2's layout NHWC and NCHW as well.
        """
        X, W, bias = inputs

        order = [val.s for val in c2_op.arg._values if val.name == "order"]
        if 1 != len(order):
            raise ValueError("Multiple order values in convolution")
        order = order[0]

        if order not in ("NHWC", "NCHW"):
            raise NotImplementedError("Unsupported order in convolution: {}", order)

        # set input axes shape
        ax_N = ng.make_axis(batch=True)
        ax_C = ng.make_axis(roles=[ar.Channel])
        ax_D = ng.make_axis(roles=[ar.Depth], length=1)
        ax_H = ng.make_axis(roles=[ar.Height])
        ax_W = ng.make_axis(roles=[ar.Width])

        # set kernel axes shape
        ax_kernel_D = ng.make_axis(roles=[ar.Depth], length=1)
        ax_kernel_H = ng.make_axis(roles=[ar.Height])
        ax_kernel_W = ng.make_axis(roles=[ar.Width])
        ax_kernel_ofm = ng.make_axis(roles=[ar.Channelout])

        # create placeholders for output axes
        oC = ng.make_axis(roles=[ar.Channel]).named('C')
        oD = ng.make_axis(roles=[ar.Depth], length=1).named('D')
        oH = ng.make_axis(roles=[ar.Height]).named('H')
        oW = ng.make_axis(roles=[ar.Width]).named('W')

        axes_order = {
            'NCHW': {'X': [ax_N, ax_C, ax_H, ax_W],
                     'W': [ax_kernel_ofm, ax_C, ax_kernel_H, ax_kernel_W]},
            'NHWC': {'X': [ax_N, ax_H, ax_W, ax_C],
                     'W': [ax_kernel_ofm, ax_kernel_H, ax_kernel_W, ax_C]},
        }

        ng.make_axes(axes_order[order]['X']).set_shape(X.axes.lengths)
        ng.make_axes(axes_order[order]['W']).set_shape(W.axes.lengths)

        if 1 != len(bias.axes):
            raise ValueError("Bias's must be 1D.")
        if ax_kernel_ofm.length != bias.axes.lengths[0]:
            raise ValueError("Bias's length must equal to number of output feature maps.")

        # strides params
        stride_size = [int(val.i) for val in c2_op.arg._values if val.name == "stride"]
        if len(stride_size) != 1:
            raise ValueError("Stride size must be scalar value")
        str_h = str_w = stride_size[0]

        # padding params
        pad_t, pad_b, pad_l, pad_r = \
            _c2_padding(c2_op,
                        in_NHWC=[ax_N.length, ax_H.length, ax_W.length, ax_C.length],
                        kernel_HWIO=[ax_kernel_H.length, ax_kernel_W.length,
                                     ax_C.length, ax_kernel_ofm.length],
                        stride_NHWC=[1, str_h, str_w, 1])

        if pad_t != pad_b or pad_l != pad_r:
            raise NotImplementedError("Requires symmetric padding in ngraph:"
                                      "pad_t(%s) == pad_b(%s) and"
                                      "pad_l(%s) == pad_r(%s)" %
                                      (pad_t, pad_b, pad_l, pad_r))

        # conv params
        params = dict(pad_d=0, pad_h=pad_t, pad_w=pad_l,
                      str_d=1, str_h=str_h, str_w=str_w,
                      dil_d=1, dil_h=1, dil_w=1)

        # input, weight, output axes
        internal_ax_dict = {
            'X': ng.make_axes([ax_C, ax_D, ax_H, ax_W, ax_N]),
            'W': ng.make_axes([ax_C, ax_kernel_D, ax_kernel_H, ax_kernel_W, ax_kernel_ofm])
        }

        oC.length = ax_kernel_ofm.length
        oH.length = output_dim(ax_H.length, ax_kernel_H.length, params['pad_h'], params['str_h'])
        oW.length = output_dim(ax_W.length, ax_kernel_W.length, params['pad_w'], params['str_w'])
        internal_ax_dict['Y'] = ng.make_axes([oC, oD, oH, oW, ax_N])

        # broadcast input / filter axes
        # flow for NHWC order:                   |  flow for NCHW order:
        # input:                                 |  input:
        #   expand dims: NHWC -> NDHWC           |    expand dims: NCHW -> NDCHW
        #   reorder:     NDHWC -> CDHWN          |    reorder:     NDCHW -> CDHWN
        # weights:                               |  weights:
        #   expand dims: (ofm)HWC -> D(ofm)HWC   |    expand dims: (ofm)CHWC -> D(ofm)CHW
        #   reorder:     D(ofm)HWC -> CDHW(ofm)  |    reorder:     D(ofm)CHW -> CDHW(ofm)

        X = ng.cast_axes(X, ng.make_axes(axes_order[order]['X']))
        X = ng.expand_dims(X, ax_D, 1)
        X = ng.axes_with_order(X, axes=internal_ax_dict['X'])
        W = ng.cast_axes(W, ng.make_axes(axes_order[order]['W']))
        W = ng.expand_dims(W, ax_kernel_D, 0)
        W = ng.axes_with_order(W, axes=internal_ax_dict['W'])

        # convolution
        Y = ng.convolution(params, X, W, axes=internal_ax_dict['Y'])

        # cast back to proper format
        Y = ng.broadcast(Y, ng.make_axes([ax_N, oD, oH, oW, oC])) if "NHWC" == order \
            else ng.broadcast(Y, ng.make_axes([ax_N, oD, oC, oH, oW]))  # NCHW

        # slice away the oD
        out_slicing = [slice(None), 0, slice(None), slice(None), slice(None)]
        Y = ng.tensor_slice(Y, out_slicing)

        def _conv_bias_add(c2_op, inputs):
            X, bias = inputs
            bias = ng.cast_axes(bias, axes=ng.make_axes([X.axes[1 if 'NCHW' == order else 3]]))
            Y = ng.Add(X, bias)
            return Y

        return _conv_bias_add(c2_op, [Y, bias])
