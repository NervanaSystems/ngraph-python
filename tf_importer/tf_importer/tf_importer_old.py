#!/usr/bin/env python
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
"""
Import a TensorFlow GraphDef from a protobuf file and convert it to neon's
computation graph.
- TODO: ops used in the CIFAR10_conv example:
    - Conv2D(tf.nn.conv2d), MaxPool(tf.nn.max_pool), LRN(tf.nn.lrn),
      BiasAdd(tf.nn.bias_add),
    - Conv2DBackpropInput, Conv2DBackpropFilter, MaxPoolGrad, LRNGrad,
      BiasAddGrad,
    - QueueDequeueMany, RandomShuffleQueue, QueneEnqueue
- TODO: ops used in the MNIST_LTSM example:
    - Sequence Comparison and Indexing: ArgMax(tf.argmax)
    - Comparison: Equal(tf.equal)
    - Shapes and Shaping: Squeeze(tf.squeeze)
    - Slicing and Joining: Slice(tf.slice), Split(tf.split), Concat(tf.concat),
      Transpose(tf.transpose)
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.stats as stats
from builtins import range

import ngraph as ng
from ngraph.op_graph.op_graph import (TensorOp, softmax, is_constant,
                                      constant_value)
from ngraph.util.generics import op_generic_method

import tensorflow as tf
from tensorflow.python.framework import tensor_util
import mimetypes
from google.protobuf import text_format

ignored_ops = {
    'ScalarSummary', 'ZerosLike', 'InTopK', 'MergeSummary',
}


def _shape_to_numeric_axis(shape):
    """
    Convert shape tuple to backend Axes objects
    Args:
        shape (tuple, list): input shapes
    Returns:
        Axes: backend axes object
    """
    if len(shape) == 0:
        return None
    axis_list = [ng.NumericAxis(s) for s in shape]
    return ng.Axes(tuple(axis_list))


def tensor_shape_to_tuple(tf_shape):
    """

    Args:
        tf_shape: TensorShape object

    Returns:
        tuple of the shape
    """
    return tuple([s.value for s in tf_shape])


def shape_to_axes(shape):
    if not shape:
        return ng.Axes()
    axes = [ng.Axis(length=s) for s in shape]
    return axes


def is_compatible_numpy_shape(left_shape, right_shape):
    if (not left_shape) or (not right_shape):
        return True
    for l, r in zip(reversed(left_shape), reversed(right_shape)):
        if l == 1 or r == 1:
            continue
        elif l != r:
            return False
    return True


def _tf_shape_to_axes(tf_shape):
    """
    Arguments:
        tf_shape: attr_value_pb2.AttrValue, tf node's shape
    Returns:
        ng.Axes
    """
    shape = [int(d.size) for d in tf_shape.shape.dim]
    axes = [ng.Axis(s) for s in shape]
    return tuple(axes)


class TFImporter:
    """
    Tensorflow GraphDef object to Neon graph converter
    Arguments:
        pb_file (str): path to protobuf file
        end_node_name (str, optional): the last node name in TensorFlow's graph
        loss_node_name (str, optional): the final node representing loss
                                        computation
        verbose (bool, optional): if True, prints TensorFlow nodes during
                                  imports`
    Attributes:
        pb_file (str): path to protobuf file
        end_node_name (str): name of the last node in TensorFlow graph
        loss_node_name (str): name of the loss node in TensorFlow graph
        verbose (bool): if True, prints TensorFlow nodes during imports
        x (Op): Op corresponding to TensorFlow's input data placeholder
        y (Op): Op corresponding to TensorFlow's training target placeholder
        init_op (Op): initialization Op for allocation / init tensors and consts
        last_op (Op): Op corresponding to `end_node_name`
        loss_op (Op): Op to calculate loss value, corresponds to
                      `loss_node_name`
        update_op (Op): Op corresponding to weight update
        variables (dict): Trainable variables
        name_to_op (dict): maps TensorFlow node name to Neon Op
        name_to_placeholders (dict0): maps TF node name to placeholders
        ignored_nodes (set): ignored TensorFlow nodes
        name_to_axes (dict): maps TensorFlow name to NumericAxis
        batch_axis (NumericAxis): the batch axis
        y_axis (NumericAxis): the y axis of result

    TODO:
        Split this class to multiple files, using dict() to store mapping for
        functions. The main importer class only does registration, while
        implementations of each function shall not have stateful information.
    """

    def __init__(self, pb_file, end_node_name="", loss_node_name="",
                 verbose=False):
        # input fields
        self.pb_file = pb_file
        self.end_node_name = end_node_name
        self.loss_node_name = loss_node_name
        self.verbose = verbose

        # special ops
        self.x = None
        self.y = None
        self.init_op = None
        self.last_op = None
        self.loss_op = None
        self.update_op = None

        # collections
        self.variables = dict()  # trainable variables
        self.name_to_op = dict()  # maps TF node name to Neon op
        self.name_to_placeholders = dict()  # maps TF node name to placeholders
        self.ignored_nodes = set()

        # read graph_def
        graph_def = tf.GraphDef()
        if mimetypes.guess_type(pb_file)[0] == 'text/plain':
            graph_def = tf.GraphDef()
            with open(pb_file, 'r') as f:
                text_format.Merge(f.read(), graph_def)
        else:
            with open(pb_file, 'rb') as f:
                graph_def.ParseFromString(f.read())

        # axis
        # self.name_to_axes, self.batch_axis, self.y_axis = _scan_axes(graph_def)

        # process nodes
        for tf_node in graph_def.node:
            if self.verbose:
                print(tf_node)
            self._process(tf_node)

        # last op and loss op
        self.last_op = self.name_to_op[tf_node.name]
        if loss_node_name in self.name_to_op:
            self.loss_op = self.name_to_op[loss_node_name]

    def _process(self, tf_node):
        """
        Process one TensorFlow node. It checks validity then calls `_convert`
        to do the conversion.
        Args:
          tf_node (NodeDef): a TensorFlow node
        """

        # skip ignored ops
        if tf_node.op in ignored_ops or 'save' in tf_node.name:
            self.ignored_nodes.add(tf_node.name)
            return

        # check if one of the inputs is ignored
        inputs = tf_node.input
        for input in inputs:
            if input in self.ignored_nodes:
                self.ignored_nodes.add(tf_node.name)
                return

        # convert other ops
        self.name_to_op[tf_node.name] = None
        self._convert(tf_node)

    @op_generic_method
    def _convert(self, tf_node):
        """
        Converts one TensorFlow node. Entry point of generic function.
        Args:
          tf_node (NodeDef): a TensorFlow node
        """
        raise NotImplementedError('op not supported')

    @_convert.on_op(['Tanh', 'Sigmoid'])
    def _convert(self, tf_node):
        """
        Tanh: `tf.tanh(x, name=None)`
            Args:
                x: A Tensor or SparseTensor with type float, double, int32,
                   complex64, int64, or qint32.
                name: A name for the operation (optional).
            Returns:
                A Tensor or SparseTensor respectively with the same type as
                x if x.dtype != qint32 otherwise the return type is quint8.

        Sigmoid: `tf.sigmoid(x, name=None)`
            Computes sigmoid of x element-wise. Specifically,
            y = 1 / (1 + exp(-x)).

            Args:
                x: A Tensor with type float32, float64, int32, complex64, int64,
                   or qint32.
                name: A name for the operation (optional).
            Returns:
                A Tensor with the same type as x if x.dtype != qint32 otherwise
                the return type is quint8.
        """
        # unary ops
        unary_ops = {
            'Tanh': ng.tanh,
            'Sigmoid': ng.sigmoid,
            # TODO: 'Relu': be.relu,
        }
        # get inputs
        left = self.name_to_op[tf_node.input[0]]

        # result
        result_op = unary_ops[tf_node.op](left)

        # save to dict
        self.name_to_op[tf_node.name] = result_op

    @_convert.on_op(['Add', 'Div', 'Maximum', 'Mul'])
    def _convert(self, tf_node):
        binary_ops = {
            'Add': ng.add,
            'Div': ng.divide,
            'Maximum': ng.maximum,
            'Mul': ng.multiply,
            # TODO: 'Mod', be.mod,
        }
        # get inputs
        left = self.name_to_op[tf_node.input[0]]
        right = self.name_to_op[tf_node.input[1]]

        # check if shape compatibility
        left_shape = left.axes.lengths
        right_shape = right.axes.lengths
        assert is_compatible_numpy_shape(left_shape, right_shape)

        if left_shape and right_shape:
            """
            Cast axes in numpy broadcast mapping rule

            1. introduce dummy length 1 axes to match left / right length
            2. keep maps for matching left / right / result axes
            3. slice left / right to remove length 1 axes if not both of them
               are length 1
            4. cast right to left by matching axes
            5. perform binary op
            6. cast and broadcast result
            """

            left_dim = len(left.axes)
            right_dim = len(right.axes)

            # pad left and right axis to be the same length, align right
            result_dim = max(left_dim, right_dim)
            left_axes_pad = [ng.Axis(length=1) for _ in
                             range(result_dim - left_dim)] + list(left.axes)
            right_axes_pad = [ng.Axis(length=1) for _ in
                              range(result_dim - right_dim)] + list(right.axes)
            result_axes = [ng.Axis(length=max(l.length, r.length)) for l, r
                           in zip(left_axes_pad, right_axes_pad)]

            # broadcast left / right, introducing dummy length 1 axes
            left = ng.Broadcast(left, axes=left_axes_pad)
            right = ng.Broadcast(right, axes=right_axes_pad)

            # make two-way map of lr matching axes and map for result axes
            lr_axes_map = dict()
            result_axes_map = dict()
            for l, r, re in zip(left.axes, right.axes, result_axes):
                lr_axes_map[l] = r
                lr_axes_map[r] = l
                result_axes_map[l] = re
                result_axes_map[r] = re

            # get left / right slice
            left_slice = []
            right_slice = []
            for l, r in zip(left.axes, right.axes):
                if l.length == 1 and r.length != 1:
                    left_slice.append(0)
                else:
                    left_slice.append(slice(None))
                if r.length == 1 and l.length != 1:
                    right_slice.append(0)
                else:
                    right_slice.append(slice(None))

            # perform slicing
            left_sliced = ng.Slice(left, left_slice)
            right_sliced = ng.Slice(right, right_slice)

            # now cast the right_sliced to left_sliced from the axis map
            right_casted_axes = []
            for r in right_sliced.axes:
                if r in lr_axes_map and lr_axes_map[r] in left_sliced.axes:
                    right_casted_axes.append(lr_axes_map[r])
                else:
                    right_casted_axes.append(r)
            right_sliced_casted = ng.AxesCastOp(right_sliced,
                                                axes=right_casted_axes)

            # perform binary op
            result_op = binary_ops[tf_node.op](left_sliced, right_sliced_casted)

            # cast result axis and broadcast to full result axes
            trimmed_result_axes = [result_axes_map[re] for re in result_op.axes]
            result_op = ng.AxesCastOp(result_op, trimmed_result_axes)
            result_op = ng.Broadcast(result_op, axes=result_axes)
        else:
            # don't need to do any axes casting
            result_op = binary_ops[tf_node.op](left, right)

        # save to dict
        self.name_to_op[tf_node.name] = result_op

        # TODO: remove this hardcoded branch after ExpandDims op is implemented
        # if tf_node.name == 'gradients/xentropy_grad/mul':
        #     # use be.Constant(1. / self.bastch_axis.length) as temporal result
        #     # to replace the output of ExpandDims  (self.name_to_op[tf_node.input[0]])
        #     self.name_to_op[tf_node.name] = binary_ops[tf_node.op](
        #         ng.Constant(1. / self.batch_axis.length),
        #         self.name_to_op[tf_node.input[1]], name=tf_node.name)

    @_convert.on_op(['MatMul'])
    def _convert(self, tf_node):
        """
        TF Docs:
            - The inputs must be two-dimensional matrices, with matching inner
              dimensions, possibly after transposition.
            - Both matrices must be of the same type. The supported types are:
              float32, float64, int32, complex64.
            - Either matrix can be transposed on the fly by setting the
              corresponding flag to True. This is False by default.
            - If one or both of the matrices contain a lot of zeros, a more
              efficient multiplication algorithm can be used by setting the
              corresponding a_is_sparse or b_is_sparse flag to True. These are
              False by default.
        TF Args:
            a: Tensor of type float32, float64, int32 or complex64.
            b: Tensor with same type as a.
            transpose_a: If True, a is transposed before multiplication.
            transpose_b: If True, b is transposed before multiplication.
            a_is_sparse: If True, a is treated as a sparse matrix.
            b_is_sparse: If True, b is treated as a sparse matrix.
            name: Name for the operation (optional).
        """
        # get inputs
        left = self.name_to_op[tf_node.input[0]]
        right = self.name_to_op[tf_node.input[1]]

        # check shape
        assert len(left.axes) == len(right.axes) == 2
        assert left.axes[1].length == right.axes[0].length

        # cast axis
        right_axes = ng.Axes([left.axes[1], right.axes[1]])
        right_casted = ng.AxesCastOp(right, axes=right_axes)

        # result op
        result_op = ng.dot(left, right_casted, name=tf_node.name)

        # save to dict
        self.name_to_op[tf_node.name] = result_op

    @_convert.on_op(['Mean', 'Sum'])
    def _convert(self, tf_node):
        reduction_ops = {
            'Mean': ng.mean,
            'Sum': ng.sum,
            # TODO: 'Prod': be.prod,
        }

        input_tensor = self.name_to_op[tf_node.input[0]]
        assert isinstance(input_tensor, TensorOp)
        input_tensor_axes = self.name_to_op[tf_node.input[0]].axes
        if self.name_to_op[tf_node.input[1]] is None:
            reduction_indices = None
        else:
            reduction_indices = self.name_to_op[tf_node.input[1]].const

        reduction_axes = ()
        if reduction_indices is not None:
            for i in reduction_indices:
                reduction_axes += (input_tensor_axes[int(i)],)

        self.name_to_op[tf_node.name] = reduction_ops[tf_node.op](
            input_tensor,
            reduction_axes=ng.Axes(reduction_axes),
            name=tf_node.name)

    @_convert.on_op('Relu')
    def _convert(self, tf_node):
        self.name_to_op[tf_node.name] = ng.maximum(
            self.name_to_op[tf_node.input[0]], 0)

    @_convert.on_op('Identity')
    def _convert(self, tf_node):
        self.name_to_op[tf_node.name] = self.name_to_op[tf_node.input[0]]

    @_convert.on_op('Placeholder')
    def _convert(self, tf_node):
        """
        TF Docs:
            - Inserts a placeholder for a tensor that will be always fed.
            - Important: This tensor will produce an error if evaluated. Its
              value must be fed using the feed_dict optional argument to
              Session.run(), Tensor.eval(), or Operation.run().
        TF Args:
            dtype: The type of elements in the tensor to be fed.
            shape: The shape of the tensor to be fed (optional). If the shape
                   is not specified, you can feed a tensor of any shape.
            name: A name for the operation (optional).
        TF Returns:
            A Tensor that may be used as a handle for feeding a value, but not
            evaluated directly.
        """
        axes = _tf_shape_to_axes(tf_node.attr['shape'])
        ng_op = ng.placeholder(axes=axes, name=tf_node.name)
        self.name_to_op[tf_node.name] = ng_op
        self.name_to_placeholders[tf_node.name] = ng_op

    @_convert.on_op('Const')
    def _convert(self, tf_node):
        """
        TF Docs:
            TensorFlow provides several operations that you can use to generate
            constants. More specifically,

                tf.zeros(shape, dtype=tf.float32, name=None)
                tf.zeros_like(tensor, dtype=None, name=None)
                tf.ones(shape, dtype=tf.float32, name=None)
                tf.ones_like(tensor, dtype=None, name=None)
                tf.fill(dims, value, name=None)
                tf.constant(value, dtype=None, shape=None, name=Const)

            They all create op: "Const".
        """
        # convert to numpy value
        np_val = tensor_util.MakeNdarray(tf_node.attr['value'].tensor)
        ng_node = ng.Constant(np_val, axes=shape_to_axes(np_val.shape))
        self.name_to_op[tf_node.name] = ng_node

    @_convert.on_op('Variable')
    def _convert(self, tf_node):
        self.name_to_op[tf_node.name] = ng.Variable(
            axes=self.name_to_axes[tf_node.name], name=tf_node.name)
        self.variables[tf_node.name] = self.name_to_op[tf_node.name]

    @_convert.on_op('Assign')
    def _convert(self, tf_node):
        var = self.name_to_op[tf_node.input[0]]
        init_value = self.name_to_op[tf_node.input[1]]
        self.name_to_op[tf_node.name] = ng.assign(var, init_value)
        var.initializers.append(self.name_to_op[tf_node.name])

    @_convert.on_op('AssignAdd')
    def _convert(self, tf_node):
        # TODO: check operations for scala variable
        # Things may broken for other graph in which the scala variable is not
        # named 'global_step'
        if tf_node.input[0] == 'global_step':
            return

        var = self.name_to_op[tf_node.input[0]]
        tensor_to_add = self.name_to_op[tf_node.input[1]]
        self.name_to_op[tf_node.name] = ng.assign(var, var + tensor_to_add)

    @_convert.on_op('Fill')
    def _convert(self, tf_node):
        # Creates a tensor filled with a scalar value.
        shape_tensor = self.name_to_op[tf_node.input[0]].const
        init_val = self.name_to_op[tf_node.input[1]]
        assert is_constant(init_val)

        if len(shape_tensor.shape) == 0:
            self.name_to_op[tf_node.name] = ng.Constant(
                constant_value(init_val), name=tf_node.name)
        else:
            shape = tuple([int(s) for s in shape_tensor])
            array = np.zeros(shape)
            array.fill(constant_value(init_val))
            axes = _shape_to_numeric_axis(shape)
            self.name_to_op[tf_node.name] = ng.Constant(array,
                                                        axes=axes,
                                                        name=tf_node.name)

    @_convert.on_op(['TruncatedNormal', 'RandomStandardNormal'])
    def _convert(self, tf_node):
        # TODO: implement tf.truncated_normal and tf.random_normal
        # get shape
        shape = self.name_to_op[tf_node.input[0]].const
        shape = tuple([int(s) for s in shape])

        if tf_node.op == 'TruncatedNormal':
            lower, upper = -2.0, 2.0
            mu, sigma = 0, 1
            X = stats.truncnorm((lower - mu) / sigma,
                                (upper - mu) / sigma, loc=mu,
                                scale=sigma)
            val = X.rvs(shape)
        elif tf_node.op == "RandomStandardNormal":
            val = -0.5 + np.random.random_sample(shape).astype(
                np.float32)

        axes = _shape_to_numeric_axis(shape)
        self.name_to_op[tf_node.name] = ng.Constant(val, axes=axes,
                                                    name=tf_node.name)

    @_convert.on_op('Cast')
    def _convert(self, tf_node):
        # TODO: need a real cast, currently just skip this op
        # dst_type = tf_node.attr['DstT']
        # src_type = tf_node.attr['SrcT']
        self.name_to_op[tf_node.name] = self.name_to_op[tf_node.input[0]]

    @_convert.on_op('SparseSoftmaxCrossEntropyWithLogits')
    def _convert(self, tf_node):
        # implementation of tf.nn.sparse_softmax_cross_entropy_with_logits
        # check its doc via https://goo.gl/7ytJNB and its C++ implementation via
        # https://goo.gl/z5T2my

        pred = softmax(self.name_to_op[tf_node.input[0]],
                       ng.Axes(self.y_axis, ))
        label = self.name_to_op[tf_node.input[1]]

        self.name_to_op[tf_node.name] = ng.cross_entropy_multi(pred, label,
                                                               out_axes=(
                                                                   self.batch_axis,))
        # equivalent: op = -be.sum(safelog(pred) * label * np.float(1. / np.log(2.0)),
        #                             out_axes=(self.bastch_axis,))

        # this op also calculates gradients and saved in the second output
        sum_exp_logits = ng.sum(pred, out_axes=(self.batch_axis,))
        grad = ng.divide(pred, sum_exp_logits) - label
        self.name_to_op[tf_node.name + ":1"] = grad

    @_convert.on_op('Prod')
    def _convert(self, tf_node):
        # TODO: implement tf.reduce_prod and merge with reduction_ops
        prod_val = np.prod(self.name_to_op[tf_node.input[0]].const)
        self.name_to_op[tf_node.name] = ng.Constant(prod_val,
                                                    name=tf_node.name)

    @_convert.on_op('Shape')
    def _convert(self, tf_node):
        axes = self.name_to_op[tf_node.input[0]].axes
        shape = [axis.length for axis in axes]

        if len(shape) == 0:
            self.name_to_op[tf_node.name] = ng.Constant(0,
                                                        name=tf_node.name)
        else:
            axes = ng.Axes(ng.NumericAxis(len(shape)), )
            self.name_to_op[tf_node.name] = ng.Constant(np.array(shape),
                                                        axes=axes,
                                                        name=tf_node.name)

    @_convert.on_op('Rank')
    def _convert(self, tf_node):
        # The rank of a tensor is the number of axis
        shape = self.name_to_op[tf_node.input[0]].shape
        self.name_to_op[tf_node.name] = ng.Constant(len(shape),
                                                    name=tf_node.name)

    @_convert.on_op('Size')
    def _convert(self, tf_node):
        axes = self.name_to_op[tf_node.input[0]].axes
        shape = [axis.length for axis in axes]
        self.name_to_op[tf_node.name] = ng.Constant(np.prod(shape),
                                                    name=tf_node.name)

    @_convert.on_op('Range')
    def _convert(self, tf_node):
        start = self.name_to_op[tf_node.input[0]]
        limit = self.name_to_op[tf_node.input[1]]
        delta = self.name_to_op[tf_node.input[2]]
        nums = np.arange(start.const, limit.const, delta.const).astype(
            np.float32)
        self.name_to_op[tf_node.name] = ng.Constant(nums, axes=ng.Axes(
            ng.NumericAxis(len(nums)), ), name=tf_node.name)

    @_convert.on_op('Mod')
    def _convert(self, tf_node):
        # TODO: implement tf.mod, currently just skip
        self.name_to_op[tf_node.name] = self.name_to_op[tf_node.input[0]]

    @_convert.on_op('DynamicStitch')
    def _convert(self, tf_node):
        # TODO: implement tf.dynamic_stich, currently just use a constant
        self.name_to_op[tf_node.name] = ng.Constant(1)

    @_convert.on_op('Reshape')
    def _convert(self, tf_node):
        # TODO: implement tf.reshape
        # Currently it just does nothing but pass the first input without
        # actually reshape
        self.name_to_op[tf_node.name] = self.name_to_op[tf_node.input[0]]

    @_convert.on_op('Tile')
    def _convert(self, tf_node):
        # Constructs a tensor by tiling a given tensor. Currently use numpy.tile
        # The first input is the result of tf.reshape, which is currently not
        # available
        # TODO: implement tf.reshape and tf.tile

        # input = self.name_to_op[tf_node.input[0]]
        # multiples = self.name_to_op[tf_node.input[1]]

        # should use the result of multiples as the second arg for np.tile
        # but the value is not available when this graph is constructed.
        array = []
        if is_constant(self.name_to_op[tf_node.input[0]]):
            array = constant_value(self.name_to_op[tf_node.input[0]])
        val = np.tile(array, self.batch_axis.length)
        shape = val.shape
        if len(shape) == 1:
            self.name_to_op[tf_node.name] = ng.Constant(val, axes=ng.Axes(
                ng.NumericAxis(shape[0]), ), name=tf_node.name)
        else:
            assert False

    @_convert.on_op('ExpandDims')
    def _convert(self, tf_node):
        # TODO: implement tf.expand_dims
        # dim = self.name_to_op[tf_node.input[1]]
        self.name_to_op[tf_node.name] = self.name_to_op[tf_node.input[0]]

    @_convert.on_op('BroadcastGradientArgs')
    def _convert(self, tf_node):
        # implementation of bcast_ops.cc (https://goo.gl/5vx4QN)
        sx = self.name_to_op[tf_node.input[0]].const
        sy = self.name_to_op[tf_node.input[1]].const

        grad_x_reduce_ = []
        grad_y_reduce_ = []

        if not np.array_equal(sx, sy):
            x = sx[::-1]
            y = sy[::-1]

            if len(x) > len(y):
                y = np.pad(y, (0, len(x) - len(y)), 'constant',
                           constant_values=1)
            else:
                x = np.pad(x, (0, len(y) - len(x)), 'constant',
                           constant_values=1)

        n = len(x)
        for i in range(n):
            if not x[i] == y[i]:
                if x[i] == 1:
                    grad_x_reduce_.append(n - 1 - i)
                elif y[i] == 1:
                    grad_y_reduce_.append(n - 1 - i)

        if grad_x_reduce_:
            val_x = np.array(grad_x_reduce_)
            axes = ng.Axes(ng.NumericAxis(len(grad_x_reduce_)), )
            self.name_to_op[tf_node.name] = ng.Constant(val_x, axes=axes,
                                                        name=tf_node.name)

        self.name_to_op[tf_node.name + ":1"] = None
        if grad_y_reduce_:
            val_y = np.array(grad_y_reduce_)
            axes = ng.Axes(ng.NumericAxis(len(grad_y_reduce_)), )
            self.name_to_op[tf_node.name + ":1"] = ng.Constant(val_y, axes=axes,
                                                               name=tf_node.name)

    @_convert.on_op('ReluGrad')
    def _convert(self, tf_node):
        gradient = self.name_to_op[tf_node.input[0]]
        output = self.name_to_op[tf_node.input[1]]
        self.name_to_op[tf_node.name] = gradient * output

    @_convert.on_op('ApplyGradientDescent')
    def _convert(self, tf_node):
        var = self.name_to_op[tf_node.input[0]]
        lr = self.name_to_op[tf_node.input[1]]
        grad = self.name_to_op[tf_node.input[2]]
        updated_var = var - lr * grad
        self.name_to_op[tf_node.name] = ng.assign(var, updated_var)

    @_convert.on_op('NoOp')
    def _convert(self, tf_node):
        # NoOp adds '^' before each original input name
        if tf_node.name == "GradientDescent/update":
            # gradient descent ops
            self.name_to_op[tf_node.name] = ng.doall(
                all=[self.name_to_op[input[1:]] for input in tf_node.input])
            self.update_op = self.name_to_op[tf_node.name]

        elif tf_node.name == "init":
            # variable initialization graph, used only once
            self.name_to_op[tf_node.name] = ng.doall(
                all=[self.name_to_op[input[1:]] for input in
                     tf_node.input[:-1]])
            self.init_op = self.name_to_op[tf_node.name]
