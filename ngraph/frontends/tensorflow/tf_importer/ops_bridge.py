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

from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.stats
import ngraph as ng
import ngraph.frontends.tensorflow.tf_importer.ngraph_shaped as ns

from ngraph.frontends.common.utils import common_conv2d_pool_padding, \
    common_conv2d_pool_output_shape
from ngraph.frontends.tensorflow.tf_importer.utils import tf_obj_shape
from ngraph.frontends.tensorflow.tf_importer.utils_pos_axes import make_pos_axes
from tensorflow.python.framework import tensor_util


def _get_reduction_indices(reduction_indices):
    try:
        assert reduction_indices.const is not None
    except:
        raise NotImplementedError(
            "[NON-NATIVE] reduction_indices be "
            "constants, cannot come from intermediate "
            "results")
    return tuple([int(ind) for ind in reduction_indices.const])


class OpsBridge(object):
    """
    Bridging op between TensorFlow / ngraph.
    """

    def __call__(self, tf_node, input_ops):
        """
        Call Op based on `tf_node.name`. Mix-in functions must have same name
        as the `tf_node.name`.

        Arguments:
            tf_node (NodeDef): a TensorFlow node
            input_ops (List): list of ngraph op

        Returns:
            The resulting ngraph op
        """
        op_name = tf_node.op

        # if op not handled, gets -1
        ng_op = getattr(self, op_name, None)

        if ng_op:
            return ng_op(tf_node, input_ops)
        else:
            # ignored op set to None
            print(tf_node.name, "ignored.")
            return None

    def Placeholder(self, tf_node, inputs):
        return ns.placeholder(shape=tf_obj_shape(tf_node), name=tf_node.name)

    def Add(self, tf_node, inputs):
        return ns.add(inputs[0], inputs[1], name=tf_node.name)

    def BiasAdd(self, tf_node, inputs):
        value, bias = inputs
        if len(bias.axes) != 1:
            raise ValueError("Bias's must be 1D.")
        if bias.axes.lengths[0] != value.axes.lengths[-1]:
            raise ValueError("Bias's length must equal to value's last dim.")
        return self.Add(tf_node, inputs)

    def Sub(self, tf_node, inputs):
        return ns.add(inputs[0], -inputs[1], name=tf_node.name)

    def Mul(self, tf_node, inputs):
        return ns.multiply(inputs[0], inputs[1], name=tf_node.name)

    def Div(self, tf_node, inputs):
        return ns.divide(inputs[0], inputs[1], name=tf_node.name)

    def Mod(self, tf_node, inputs):
        return ns.mod(inputs[0], inputs[1], name=tf_node.name)

    def Maximum(self, tf_node, inputs):
        return ns.maximum(inputs[0], inputs[1], name=tf_node.name)

    def Const(self, tf_node, inputs):
        # convert to numpy value
        np_val = tensor_util.MakeNdarray(tf_node.attr['value'].tensor)
        if np_val.dtype == np.dtype('O'):
            return None
        else:
            return ns.constant(np_val, name=tf_node.name)

    def Fill(self, tf_node, inputs):
        # get inputs
        shape_op, const_val_op = inputs

        # get shape, const_val
        shape = tuple(shape_op.const.astype(int))
        const_val = const_val_op.const

        # convert to numpy value
        np_val = np.zeros(shape)
        np_val.fill(const_val)

        # create op
        return ns.constant(np_val, name=tf_node.name)

    def TruncatedNormal(self, tf_node, inputs):
        """
        Outputs random values from a truncated normal distribution.
        `tf.truncated_normal()` call generates several ops, the
        The `TruncatedNormal` op is what we implement here.

        shape --> TruncatedNormal
                       |
                       V
        stddev -----> Mul
                       |
                       V
        mean -------> Add
                       |
                       V
                    (output)
        """
        # get inputs
        shape = tuple(inputs[0].const.astype(int))

        # generate truncated standard normal
        mu, sigma, lo, up = 0., 1., -2., 2
        generator = scipy.stats.truncnorm(
            (lo - mu) / sigma, (up - mu) / sigma, loc=mu, scale=sigma)
        np_val = generator.rvs(shape)
        return ns.constant(np_val, name=tf_node.name)

    def RandomStandardNormal(self, tf_node, inputs):
        """
        Outputs random values from a normal distribution. `tf.random_normal()`
        call generates several ops. The `RandomStandardNormal` op is what we
        implement here.

        Inputs to tf_node:
            shape, mean, dtype, seed, name
        """
        # get inputs
        shape = tuple(inputs[0].const.astype(int))

        # generate standard normal
        np_val = np.random.standard_normal(size=shape)
        return ns.constant(np_val, name=tf_node.name)

    def ZerosLike(self, tf_node, inputs):
        shape = inputs[0].axes.lengths
        np_val = np.zeros(shape)
        return ns.constant(np_val, name=tf_node.name)

    def MatMul(self, tf_node, inputs):
        """
        Inputs to tf_node:
            a, b, transpose_a, transpose_b, a_is_sparse, b_is_sparse, name
        """
        return ns.matmul(inputs[0], inputs[1],
                         transpose_a=tf_node.attr['transpose_a'].b,
                         transpose_b=tf_node.attr['transpose_b'].b,
                         name=tf_node.name)

    def Sum(self, tf_node, inputs):
        return ns.reduce_sum(inputs[0],
                             axis=_get_reduction_indices(inputs[1]),
                             name=tf_node.name)

    def Mean(self, tf_node, inputs):
        return ns.reduce_mean(inputs[0],
                              axis=_get_reduction_indices(inputs[1]),
                              name=tf_node.name)

    def Prod(self, tf_node, inputs):
        return ns.reduce_prod(inputs[0],
                              axis=_get_reduction_indices(inputs[1]),
                              name=tf_node.name)

    def Tanh(self, tf_node, inputs):
        return ns.tanh(inputs[0], name=tf_node.name)

    def Sigmoid(self, tf_node, inputs):
        return ns.sigmoid(inputs[0], name=tf_node.name)

    def Relu(self, tf_node, inputs):
        return ns.relu(inputs[0], name=tf_node.name)

    def Identity(self, tf_node, inputs):
        # TODO: currently only a pass through
        return inputs[0]

    def Log(self, tf_node, inputs):
        return ns.log(inputs[0], name=tf_node.name)

    def Neg(self, tf_node, inputs):
        return ns.negative(inputs[0], name=tf_node.name)

    def Square(self, tf_node, inputs):
        return ns.square(inputs[0], name=tf_node.name)

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
            axes = make_pos_axes(tf_obj_shape(tf_node.attr['shape']))
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

        return ng.assign(ref, value)

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
        N = ng.make_axis(name='N')
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
        # KMPQN -> NMPQK
        output = ng.axes_with_order(output, ng.make_axes([N, M, P, Q, K]))
        # NMPQK -> NPQK
        output = ng.tensor_slice(output, [slice(None), 0, slice(None),
                                          slice(None), slice(None)])

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
        N = ng.make_axis(name='N')
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
        # KMPQN -> NMPQK
        output = ng.axes_with_order(output, ng.make_axes(
            [N, M, P, Q, K]))
        # NMPQK -> NPQK
        output = ng.tensor_slice(output, [slice(None), 0, slice(None),
                                          slice(None), slice(None)])

        return output

    def SparseSoftmaxCrossEntropyWithLogits(self, tf_node, inputs):
        """
                (    N,     Y),         (    N)
        logits: (pos_1, pos_0), labels: (pos_0)
        """
        logits, labels = inputs
        return ns.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                           logits=logits,
                                                           name=tf_node.name)

    def Softmax(self, tf_node, inputs):
        # TODO: only support tf.nn.softmax(logits, dim=-1) now, should add more
        return ns.softmax(inputs[0], name=tf_node.name)

    def Rank(self, tf_node, inputs):
        """
        Returns the rank of a tensor.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            input, name
        """
        # get inputs
        left = inputs[0]

        # get rank
        try:
            rank = len(left.axes.lengths)
        except:
            raise NotImplementedError("[NON-NATIVE] `Rank` op's axes must be "
                                      "pre-determined before execution.")
        # return
        return ng.constant(rank, ng.make_axes([])).named(tf_node.name)

    def Range(self, tf_node, inputs):
        """
        Creates a sequence of integers.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            start, limit, delta, name
        """
        # get inputs
        start, limit, delta = inputs

        # get range
        try:
            range_val = np.arange(start.const, limit.const, delta.const)
        except:
            raise NotImplementedError("[NON-NATIVE] Input to `Range` must all "
                                      "be integer, dynamic allocation is not "
                                      "supported.")

        # return
        return ng.constant(range_val,
                           make_pos_axes(range_val.shape)).named(tf_node.name)

    def Size(self, tf_node, inputs):
        """
        Returns the size of a tensor.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            input, name
        """
        # get inputs
        left = inputs[0]

        # get rank
        try:
            size = np.prod(left.axes.lengths)
        except:
            raise NotImplementedError("[NON-NATIVE] `Size` op's axes must be "
                                      "pre-determined before execution.")
        # return
        return ng.constant(size, ng.make_axes([])).named(tf_node.name)

    def Cast(self, tf_node, inputs):
        """
        Casts a tensor to a new type.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            x, dtype, name
        """
        # TODO: now only a pass through
        # get src and dst datatypes
        # dst_type = tf_node.attr['DstT']
        # src_type = tf_node.attr['SrcT']
        return inputs[0]

    def Shape(self, tf_node, inputs):
        """
        Returns the shape of a tensor.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            input, name
        """

        # get inputs
        left = inputs[0]

        # get shape
        try:
            shape = left.axes.lengths
        except:
            raise NotImplementedError("[NON-NATIVE] `Size` op's axes must be "
                                      "pre-determined before execution.")
        axes = ng.make_axes([ng.make_axis(len(left.axes.lengths)), ])

        # return
        return ng.constant(shape, axes).named(tf_node.name)

    def Reshape(self, tf_node, inputs):
        """
        Reshapes a tensor.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            tensor, shape, name
        """
        # TODO: currently only support constants and flatten to 1d and 2d
        # get inputs
        tensor, shape = inputs

        def get_flatten_idx(shape_i, shape_o):
            """
            check if flattening shape is valid
            Args:
                shape_i: input tensor shape
                shape_o: output flattend tensor shape

            Returns:
                None if flatten not valid, otherwise the flatten_at index
            """
            return None

        # get input and output shape
        shape_i = tensor.shape.lengths
        shape_o = tuple(shape.const.astype(int))
        if np.prod(shape_i) != np.prod(shape_o):
            raise ValueError("Total size of input and output dimension "
                             "mismatch.")

        if tensor.const is not None:
            # reshape const
            np_val = np.reshape(tensor.const, shape_o)
            return ng.constant(np_val,
                               make_pos_axes(np_val.shape)).named(tf_node.name)
        else:
            ndims_o = len(shape_o)
            if ndims_o != 1 and ndims_o != 2:
                raise NotImplementedError("Reshape can only support flatten"
                                          "to 1d or 2d.")
            if ndims_o == 1:
                tensor = ng.flatten(tensor)
            else:
                cumprods = list(np.cumprod(shape_i))
                flatten_at_idx = cumprods.index(shape_o[0]) + 1
                tensor = ng.flatten_at(tensor, flatten_at_idx)
            res = ng.cast_axes(tensor, make_pos_axes(shape_o))
            return res.named(tf_node.name)

    def Tile(self, tf_node, inputs):
        """
        Constructs a tensor by tiling a given tensor.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            tensor, shape, name
        """
        tensor, multiples = inputs

        # get inputs
        try:
            input_val = tensor.const
            multiples_val = multiples.const
        except:
            raise NotImplementedError(
                "Tile not supported in ngraph, "
                "currently only const tensor is supported.")

        # check shapes
        input_shape = input_val.shape
        input_ndims = len(input_shape)
        assert input_ndims >= 1 and input_ndims == len(multiples_val)

        output_val = np.tile(input_val, multiples_val.astype(int))

        # make new constants
        return ng.constant(output_val,
                           make_pos_axes(output_val.shape)).named(tf_node.name)

    def ExpandDims(self, tf_node, inputs):
        """
        Inserts a dimension of 1 into a tensor's shape.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            input, dim, name
        """
        # get input
        tensor, dim = inputs[0], int(inputs[1].const)

        # check `-1-input.dims() <= dim <= input.dims()`
        input_ndims = len(tensor.axes.lengths)
        assert -1 - input_ndims <= dim <= input_ndims

        # deal with negative number
        if dim < 0:
            dim = input_ndims + 1 + dim

        # create new axis
        one_axis = ng.make_axis(length=1)

        # get output axis
        pre_axis = [axis for axis in tensor.axes[:dim]]  # avoid FlattenedAxis
        pos_axis = [axis for axis in tensor.axes[dim:]]  # avoid FlattenedAxis
        out_axis = ng.make_axes(pre_axis + [one_axis] + pos_axis)

        # broadcast
        return ng.broadcast(tensor, out_axis)
