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
from __future__ import division
import ngraph as ng
import math
from ngraph.frontends.tensorflow.tf_importer.ops_base import OpsBase
from ngraph.frontends.neon import ar
from ngraph.frontends.neon.layer import output_dim


def tf_conv2d_pool_output_shape(input_shape, filter_shape, strides, padding):
    """
    Get tensorflow's tf.nn.conv2d output shape
    TODO: currently only support NHWC * RSCK, to support NCHW.

    Args:
        input_shape: [batch, in_height, in_width, in_channels].
        filter_shape: [filter_height, filter_width, in_channels, out_channels].
        strides: List of ints of length 4.
        padding: A string from: "SAME", "VALID".

    Returns:
        output shape of tf.nn.conv2d
    """
    # check inputs
    if padding != 'SAME' and padding != 'VALID':
        raise ValueError("Padding must be 'SAME' or 'valid'.")
    if not (len(input_shape) == len(filter_shape) == len(strides) == 4):
        raise ValueError(
            "input_shape, filter_shape, strides must be length 4.")

    # get input / filter shape
    N, H, W, C = input_shape
    R, S, C_, K = filter_shape
    if C != C_:
        raise ValueError("Input channel must be the same as filter channel.")

    # only support [1, X, X, 1] strides for importer now
    if strides[0] != 1 or strides[3] != 1:
        raise NotImplementedError("Strides on batch axis (N) and channel axis "
                                  "(C) must be 1 for importer.")

    # get output shape
    if padding == 'SAME':
        out_height = math.ceil(float(H) / float(strides[1]))
        out_width = math.ceil(float(W) / float(strides[2]))
    else:
        out_height = math.ceil(float(H - R + 1) / float(strides[1]))
        out_width = math.ceil(float(W - S + 1) / float(strides[2]))

    return (N, out_height, out_width, K)


def tf_conv2d_pool_padding(input_shape, filter_shape, strides, padding):
    """
    Get tensorflow's tf.nn.conv2d padding size
    TODO: currently only support NHWC * RSCK, to support NCHW.

    Args:
        input_shape: [batch, in_height, in_width, in_channels].
        filter_shape: [filter_height, filter_width, in_channels, out_channels].
        strides: List of ints of length 4.
        padding: A string from: "SAME", "VALID".

    Returns:
        pad_top, pad_bottom, pad_left, pad_right
    """
    # check validity and get output size
    _, out_height, out_width, _ = tf_conv2d_pool_output_shape(
        input_shape, filter_shape, strides, padding)
    if padding == 'SAME':
        # get input / filter shape
        N, H, W, C = input_shape
        R, S, C_, K = filter_shape

        # get padding size
        pad_along_height = ((out_height - 1) * strides[1] + R - H)
        pad_along_width = ((out_width - 1) * strides[2] + S - W)
        pad_top = int(pad_along_height) // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = int(pad_along_width) // 2
        pad_right = pad_along_width - pad_left
        return (pad_top, pad_bottom, pad_left, pad_right)
    else:
        return (0, 0, 0, 0)


class OpsNN(OpsBase):
    """
    Mix-in class for tf.nn related ops
    """

    def Conv2D(self, tf_node, inputs):
        """
        Computes a 2-D convolution given 4D input and filter tensors.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            input, filter

        TODO: assume default tensorflow layout NHWC, RSCK,
              need to support NCHW as well
              need to clean up / merge with maxpool

        Notes on output shape:
            https://www.tensorflow.org/api_docs/python/nn.html#convolution
        """
        image, weight = inputs

        # TODO: currently NHWC only
        assert tf_node.attr['data_format'].s.decode("ascii") == "NHWC"

        # set axes shape
        ax_N = ng.make_axis(batch=True)
        ax_C = ng.make_axis(roles=[ar.features_input])
        ax_D = ng.make_axis(roles=[ar.features_0], length=1)
        ax_H = ng.make_axis(roles=[ar.features_1])
        ax_W = ng.make_axis(roles=[ar.features_2])

        ax_T = ng.make_axis(roles=[ar.features_0], length=1)
        ax_R = ng.make_axis(roles=[ar.features_1])
        ax_S = ng.make_axis(roles=[ar.features_2])
        ax_K = ng.make_axis(roles=[ar.features_output])

        oC = ng.make_axis(name='C', roles=[ar.features_input])
        oD = ng.make_axis(name='D', roles=[ar.features_0], length=1)
        oH = ng.make_axis(name='H', roles=[ar.features_1])
        oW = ng.make_axis(name='W', roles=[ar.features_2])

        ng.make_axes([ax_N, ax_H, ax_W, ax_C]).set_shape(image.axes.lengths)
        ng.make_axes([ax_R, ax_S, ax_C, ax_K]).set_shape(weight.axes.lengths)

        # strides params
        tf_strides = [int(s) for s in list(tf_node.attr['strides'].list.i)]
        if len(tf_strides) != 4:
            raise ValueError("Length of strides my be 4.")
        if tf_strides[0] != 1:
            raise NotImplementedError('Strides on batch axis (N) must be 1.')
        if tf_strides[3] != 1:
            raise NotImplementedError('Strides on channel axis (C) must be 1.')
        str_h, str_w = tf_strides[1], tf_strides[2]

        # padding params
        padding = tf_node.attr['padding'].s.decode("ascii")
        pad_t, pad_b, pad_l, pad_r = tf_conv2d_pool_padding(
            image.axes.lengths, weight.axes.lengths, tf_strides, padding)
        if pad_t != pad_b or pad_l != pad_r:
            raise NotImplementedError("Requires symmetric padding in ngraph:"
                                      "pad_t(%s) == pad_b(%s) and"
                                      "pad_l(%s) == pad_r(%s)" %
                                      (pad_t, pad_b, pad_l, pad_r))

        # conv params
        params = dict(pad_d=0, pad_h=pad_t, pad_w=pad_l,
                      str_d=1, str_h=str_h, str_w=str_w,
                      fil_d=1, dil_h=1, dil_w=1)

        # i, f, o axes
        ax_i = ng.make_axes([ax_C, ax_D, ax_H, ax_W, ax_N])
        ax_f = ng.make_axes([ax_C, ax_T, ax_R, ax_S, ax_K])
        ax_o = ng.make_axes([oC, oD, oH, oW, ax_N])

        oC.length = ax_K.length
        oH.length = output_dim(ax_H.length, ax_R.length, pad_t, str_h)
        oW.length = output_dim(ax_W.length, ax_S.length, pad_l, str_w)

        # broadcast input / filter axes
        image = ng.cast_axes(image, ng.make_axes([ax_N, ax_H, ax_W, ax_C]))
        image = ng.expand_dims(image, ax_D, 1)  # NHWC -> NDHWC
        image = ng.axes_with_order(image, axes=ax_i)  # NDHWC -> CDHWN
        weight = ng.cast_axes(weight, ng.make_axes([ax_R, ax_S, ax_C, ax_K]))
        weight = ng.expand_dims(weight, ax_T, 0)  # RSCK -> TRSCK
        weight = ng.axes_with_order(weight, axes=ax_f)  # TRSCK -> CTRSK

        # convolution
        output = ng.convolution(params, image, weight, axes=ax_o)

        # cast back to NHWC
        oC, oD, oH, oW, oN = output.axes
        output = ng.broadcast(output, ng.make_axes([oN, oD, oH, oW, oC]))

        # slice away the oD
        out_slicing = [slice(None), 0, slice(None), slice(None), slice(None)]
        output = ng.tensor_slice(output, out_slicing)

        return output

    def MaxPool(self, tf_node, inputs):
        """
        Performs the max pooling on the input.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            input

        TODO: assume default tensorflow layout NHWC, RSCK,
              need to support NCHW as well
              need to clean up / merge with conv2d

        Notes on output shape:
            https://www.tensorflow.org/api_docs/python/nn.html#convolution
        """
        image = inputs[0]

        # TODO: currently NHWC only
        assert tf_node.attr['data_format'].s.decode("ascii") == "NHWC"

        # set axes shape
        ax_N = ng.make_axis(batch=True)
        ax_C = ng.make_axis(roles=[ar.features_input])
        ax_D = ng.make_axis(roles=[ar.features_0], length=1)
        ax_H = ng.make_axis(roles=[ar.features_1])
        ax_W = ng.make_axis(roles=[ar.features_2])

        oC = ng.make_axis(name='C', roles=[ar.features_input])
        oD = ng.make_axis(name='D', roles=[ar.features_0], length=1)
        oH = ng.make_axis(name='H', roles=[ar.features_1])
        oW = ng.make_axis(name='W', roles=[ar.features_2])

        ng.make_axes([ax_N, ax_H, ax_W, ax_C]).set_shape(image.axes.lengths)

        # ksize params
        tf_ksize = [int(s) for s in list(tf_node.attr['ksize'].list.i)]
        if len(tf_ksize) != 4:
            raise ValueError("Length of ksize my be 4.")
        if tf_ksize[0] != 1:
            raise NotImplementedError('Ksize on batch axis (N) must be 1.')
        if tf_ksize[3] != 1:
            raise NotImplementedError('Ksize on channel axis (C) must be 1.'
                                      'Cross map pooling to be implemented.')
        R, S = tf_ksize[1:3]
        T = J = 1

        # strides params
        tf_strides = [int(s) for s in list(tf_node.attr['strides'].list.i)]
        if len(tf_strides) != 4:
            raise ValueError("Length of strides my be 4.")
        if tf_strides[0] != 1:
            raise NotImplementedError('Strides on batch axis (N) must be 1.')
        if tf_strides[3] != 1:
            raise NotImplementedError('Strides on channel axis (C) must be 1.')
        str_h, str_w = tf_strides[1], tf_strides[2]

        # padding params
        padding = tf_node.attr['padding'].s.decode("ascii")
        pad_t, pad_b, pad_l, pad_r = tf_conv2d_pool_padding(
            image.axes.lengths, (R, S, ax_C.length, ax_C.length), tf_strides,
            padding)
        if pad_t != pad_b or pad_l != pad_r:
            raise NotImplementedError("Requires symmetric padding in ngraph:"
                                      "pad_t(%s) == pad_b(%s) and"
                                      "pad_l(%s) == pad_r(%s)" %
                                      (pad_t, pad_b, pad_l, pad_r))
        # pooling params
        params = dict(op='max',
                      pad_d=0, pad_h=pad_t, pad_w=pad_l, pad_c=0,
                      str_d=1, str_h=str_h, str_w=str_w, str_c=1,
                      J=J, T=T, R=R, S=S)

        oC.length = output_dim(ax_C.length, J, 0, 1)
        oH.length = output_dim(ax_H.length, R, pad_t, str_h)
        oW.length = output_dim(ax_W.length, S, pad_l, str_w)

        # i, o axes
        ax_i = ng.make_axes([ax_C, ax_D, ax_H, ax_W, ax_N])
        ax_o = ng.make_axes([oC, oD, oH, oW, ax_N])

        # broadcast input / filter axes
        image = ng.cast_axes(image, ng.make_axes([ax_N, ax_H, ax_W, ax_C]))
        image = ng.expand_dims(image, ax_D, 1)  # NHWC -> NDHWC
        image = ng.axes_with_order(image, axes=ax_i)  # NDHWC -> CDHWN

        # pooling
        output = ng.pooling(params, image, axes=ax_o)

        # cast back to NHWC
        output = ng.broadcast(output, ng.make_axes([ax_N, oD, oH, oW, oC]))

        # slice away the oD
        out_slicing = [slice(None), 0, slice(None), slice(None), slice(None)]
        output = ng.tensor_slice(output, out_slicing)

        return output

    def SparseSoftmaxCrossEntropyWithLogits(self, tf_node, inputs):
        """
        Computes softmax cross entropy. The inputs `logits` are unscaled log
        probabilities, and each row of `labels[i]` must be a valid distribution.
        Reference: https://goo.gl/z5T2my

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            logits, labels, name
        """

        # logits: (N1, Y1), labels: (N2,)
        logits, labels = inputs

        # check input dimension
        try:
            assert len(logits.axes) == 2
            assert len(labels.axes) == 1
            assert logits.axes[0].length == labels.axes[0].length
        except:
            raise NotImplementedError("logits' shape must be (Y, N), "
                                      "labels' shape must be (N,), "
                                      "other shapes not supported yet.")
        # get axis
        axis_y = logits.axes[1]

        # labels_one_hot: (Y2, N2)
        labels_one_hot = ng.one_hot(labels, axis=axis_y)

        # predicts: (N1, Y1)
        predicts = ng.softmax(logits, normalization_axes=axis_y)

        # dim-shuffle / cast to (Y1, N1)
        predicts_axes = ng.make_axes(
            [axis for axis in reversed(predicts.axes)])
        predicts = ng.axes_with_order(predicts, axes=predicts_axes)
        labels_one_hot = ng.cast_axes(labels_one_hot, predicts_axes)

        # cross_entropy: (N1,)
        cross_entropy = ng.cross_entropy_multi(
            predicts, labels_one_hot, out_axes=(logits.axes[0],))

        return cross_entropy

    def Softmax(self, tf_node, inputs):
        """
        Computes softmax activations.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            logits, name
        """
        # TODO: only support tf.nn.softmax(logits, dim=-1) now, should add more
        logits = inputs[0]
        return ng.softmax(logits, normalization_axes=logits.axes[1])
