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
from ngraph.frontends.tensorflow.tf_importer.ops_base import OpsBase
from ngraph.frontends.common.utils import common_conv2d_pool_padding,\
    common_conv2d_pool_output_shape


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

        Axes:
                      Tensorflow          Ngraph
            in       (N, H, W, C)     (C, D, H, W, N)
            filter   (R, S, C, K)     (C, T, R, S, K)
            out      (N, P, Q, K)     (K, M, P, Q, N)

        Notes on output shape:
            https://www.tensorflow.org/api_docs/python/nn.html#convolution
        """
        image, weight = inputs

        # TODO: currently NHWC only
        if tf_node.attr['data_format'].s.decode("ascii") != "NHWC":
            raise NotImplementedError("Only supports NHWC import for now.")

        # check in_C == f_C
        if image.axes.lengths[3] != weight.axes.lengths[2]:
            raise ValueError("Image's C dimension (%s) must be equal to "
                             "filter's C dimension (%s)."
                             % (image.axes.lengths[3], weight.axes.lengths[2]))

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
        pad_t, pad_b, pad_l, pad_r = common_conv2d_pool_padding(
            image.axes.lengths, weight.axes.lengths, tf_strides, padding)
        if pad_t != pad_b or pad_l != pad_r:
            raise NotImplementedError("Requires symmetric padding in ngraph:"
                                      "pad_t(%s) == pad_b(%s) and"
                                      "pad_l(%s) == pad_r(%s)" %
                                      (pad_t, pad_b, pad_l, pad_r))

        # conv params
        params = dict(pad_d=0, pad_h=pad_t, pad_w=pad_l,
                      str_d=1, str_h=str_h, str_w=str_w,
                      dil_d=1, dil_h=1, dil_w=1)

        # new axes
        C, D, H, W, T, R, S, K, M, P, Q = [ng.make_axis() for _ in range(11)]
        N = ng.make_axis(batch=True)
        D.length, T.length, M.length = 1, 1, 1  # only supports 2D conv for now

        # tf's i, f, o axes
        ax_i_tf = ng.make_axes([N, H, W, C])
        ax_f_tf = ng.make_axes([R, S, C, K])
        ax_o_tf = ng.make_axes([N, P, Q, K])
        ax_i_tf.set_shape(image.axes.lengths)
        ax_f_tf.set_shape(weight.axes.lengths)
        ax_o_tf.set_shape(common_conv2d_pool_output_shape(image.axes.lengths,
                                                          weight.axes.lengths,
                                                          tf_strides, padding))

        # ngraph's i, f, o axes
        ax_i = ng.make_axes([C, D, H, W, N])
        ax_f = ng.make_axes([C, T, R, S, K])
        ax_o = ng.make_axes([K, M, P, Q, N])

        # image NHWC -> CDHWN
        image = ng.cast_axes(image, ng.make_axes([N, H, W, C]))
        image = ng.expand_dims(image, D, 1)  # NHWC -> NDHWC
        image = ng.axes_with_order(image, ax_i)  # NDHWC -> CDHWN

        # weights RSCK -> CTRSK
        weight = ng.cast_axes(weight, ng.make_axes([R, S, C, K]))
        weight = ng.expand_dims(weight, T, 0)  # RSCK -> TRSCK
        weight = ng.axes_with_order(weight, ax_f)  # TRSCK -> CTRSK

        # convolution
        output = ng.convolution(params, image, weight, axes=ax_o)

        # output KMPQN -> NPQK
        output = ng.axes_with_order(output, ng.make_axes(
            [N, M, P, Q, K]))  # KMPQN -> NMPQK
        output = ng.tensor_slice(output, [slice(None), 0, slice(None),
                                          slice(None), slice(None)])  # NMPQK -> NPQK

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

        Axes:
                      Tensorflow          Ngraph
            in       (N, H, W, C)     (C, D, H, W, N)
            out      (N, P, Q, K)     (K, M, P, Q, N)

        Notes on output shape:
            https://www.tensorflow.org/api_docs/python/nn.html#convolution
        """
        image = inputs[0]

        # TODO: currently NHWC only
        assert tf_node.attr['data_format'].s.decode("ascii") == "NHWC"

        # new axes
        C, D, H, W, K, M, P, Q = [ng.make_axis() for _ in range(8)]
        N = ng.make_axis(batch=True)
        D.length, M.length = 1, 1  # only supports 2D conv for now

        # tf's input axes
        ax_i_tf = ng.make_axes([N, H, W, C])
        ax_i_tf.set_shape(image.axes.lengths)

        # ksize params
        tf_ksize = [int(s) for s in list(tf_node.attr['ksize'].list.i)]
        if len(tf_ksize) != 4:
            raise ValueError("Length of ksize my be 4.")
        if tf_ksize[0] != 1:
            raise NotImplementedError('Ksize on batch axis (N) must be 1.')
        if tf_ksize[3] != 1:
            raise NotImplementedError('Ksize on channel axis (C) must be 1.'
                                      'Cross map pooling to be implemented.')
        R_length, S_length = tf_ksize[1:3]
        T_length = J_length = 1

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
        pad_t, pad_b, pad_l, pad_r = common_conv2d_pool_padding(
            image.axes.lengths, (R_length, S_length, C.length, C.length),
            tf_strides, padding)
        if pad_t != pad_b or pad_l != pad_r:
            raise NotImplementedError("Requires symmetric padding in ngraph:"
                                      "pad_t(%s) == pad_b(%s) and"
                                      "pad_l(%s) == pad_r(%s)" %
                                      (pad_t, pad_b, pad_l, pad_r))
        # pooling params
        params = dict(op='max',
                      pad_d=0, pad_h=pad_t, pad_w=pad_l, pad_c=0,
                      str_d=1, str_h=str_h, str_w=str_w, str_c=1,
                      J=J_length, T=T_length, R=R_length, S=S_length)

        # tf's output axes
        ax_o_tf = ng.make_axes([N, P, Q, K])
        ax_o_tf.set_shape(common_conv2d_pool_output_shape(image.axes.lengths,
                                                          (R_length, S_length,
                                                           C.length, C.length),
                                                          tf_strides, padding))

        # ngraph's i, f, o axes
        ax_i = ng.make_axes([C, D, H, W, N])
        ax_o = ng.make_axes([K, M, P, Q, N])

        # image NHWC -> CDHWN
        image = ng.cast_axes(image, ng.make_axes([N, H, W, C]))
        image = ng.expand_dims(image, D, 1)  # NHWC -> NDHWC
        image = ng.axes_with_order(image, ax_i)  # NDHWC -> CDHWN

        # pooling
        output = ng.pooling(params, image, axes=ax_o)

        # output KMPQN -> NPQK
        output = ng.axes_with_order(output, ng.make_axes(
            [N, M, P, Q, K]))  # KMPQN -> NMPQK
        output = ng.tensor_slice(output, [slice(None), 0, slice(None),
                                          slice(None),
                                          slice(None)])  # NMPQK -> NPQK

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
