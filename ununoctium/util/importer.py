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
import a TensorFlow GraphDef from a protobuf file and convert it to Neon's computation graph.

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.stats as stats
from builtins import range
from builtins import str

from geon.frontends.neon import *

import geon as be
from geon.op_graph.arrayaxes import AxisVar
from geon.op_graph.op_graph import Tensor, softmax

import tensorflow as tf
from tensorflow.python.framework import tensor_util

# known TF operators that can be processed by Neon graph importer
known_ops = [
    'Add', 'Div', 'MatMul', 'Maximum', 'Mul', 'Mod',
    'Mean', 'Prod', 'Sum',  # Reduction
    'Relu', 'Tanh',  # Activation
    'Variable', 'Placeholder',
    'Const', 'Fill',  # Constant Value Tensors
    'Range', # Sequences
    'Assign', 'AssignAdd',
    'Cast',  # Casting
    'SparseSoftmaxCrossEntropyWithLogits',  # Classification
    'Shape', 'Rank', 'Size', 'Reshape', 'ExpandDims',  # Shapes and Shaping
    'TruncatedNormal', 'RandomStandardNormal',  # Random Tensors
    'Tile', 'DynamicStitch',  # Slicing and Joining
    'BroadcastGradientArgs', 'ApplyGradientDescent', 'ReluGrad',
    'Identity', 'NoOp',  # Control Flow
]

two_inputs_ops = {
    'Add': be.add,
    'Div': be.divide,
    'MatMul': be.dot,
    'Maximum': be.maximum,
    'Mul': be.multiply,
    # TODO: 'Mod', be.mod,
}

reduction_ops = {
    'Mean': be.mean,
    'Sum': be.sum,
    # TODO: 'Prod': be.prod,
}

one_inputs_ops = {
    'Tanh': be.tanh,
    'Sigmod': be.sig,
    # TODO: 'Relu': be.relu,
}

ignored_ops = {
    'ScalarSummary', 'ZerosLike', 'InTopK', 'MergeSummary',
}

"""
TODO: ops used in the CIFAR10_conv example:

- Conv2D(tf.nn.conv2d), MaxPool(tf.nn.max_pool), LRN(tf.nn.lrn), BiasAdd(tf.nn.bias_add),
- Conv2DBackpropInput, Conv2DBackpropFilter, MaxPoolGrad, LRNGrad, BiasAddGrad,
- QueueDequeueMany, RandomShuffleQueue, QueneEnqueue


TODO: ops used in the MNIST_LSTM example:

- Sequence Comparision and Indexing: ArgMax(tf.argmax)
- Comparison: Equal(tf.equal)
- Shapes and Shaping: Squeeze(tf.squeeze)
- Slicing and Joining: Slice(tf.slice), Split(tf.split), Concat(tf.concat), Transpose(tf.transpose)
"""


def scan_nameable_axes(graph_def, env):
    """
    [Deprecated] Scan the graph to get the nameable axes for each variable/placehoder/const.
    Variables are defined and initialized in the next round of graph traversal.

    :param
      - graph_def: a GraphDef object
    :return:
      - names_to_axes: a map from variable name to its axes
      - batch_axis: the batch axis
      - y_axis: axis for labels, not used for inference graph
    """
    name_to_axes = {}
    in_axis = None
    batch_axis = None
    y_axis = None
    y_name = ""

    for node in graph_def.node:
        inputs = []
        for i, input_name in enumerate([x for x in node.input]):
            inputs.append(input_name)

        op_type = node.op

        with be.bound_environment(env):
            if op_type == 'Placeholder':
                dims = node.attr['shape'].shape
                shape = [d.size for d in dims.dim]

                if batch_axis is None:
                    batch_axis = be.AxisVar(name='batch', length=shape[0])

                if len(shape) == 2:
                    x_axis = AxisVar(name='x', length=shape[1])
                    name_to_axes[node.name] = (x_axis, batch_axis)
                    in_axis = x_axis

                elif len(shape) == 1:
                    name_to_axes[node.name] = (AxisVar(name='y', length=10), batch_axis)
                    y_name = node.name

            elif op_type == 'Variable':
                dims = node.attr['shape'].shape
                shape = [d.size for d in dims.dim]

                if len(shape) == 2:
                    if 'weights' in node.name:
                        assert (in_axis is not None)
                        assert (in_axis.length == shape[0])
                        out_axis = AxisVar(name=node.name, length=shape[1])
                        name_to_axes[node.name] = (in_axis, out_axis)
                        in_axis = out_axis  # now the output axis becomes input axis for the next layer
                        y_axis = out_axis

                elif len(shape) == 1:
                    if 'biases' in node.name:
                        assert (in_axis is not None)
                        assert (in_axis.length == shape[0])
                        name_to_axes[node.name] = (in_axis,)

                elif len(shape) == 0:
                    name_to_axes[node.name] = (AxisVar(name=node.name, length=1),)

            elif op_type == 'Const':
                # in the frozen graph, all variables are converted to constant
                const_tensor = node.attr['value'].tensor
                shape = [d.size for d in const_tensor.tensor_shape.dim]

                if len(shape) == 1 and 'biases' in node.name:
                    assert (in_axis is not None)
                    assert (in_axis.length == shape[0])
                    name_to_axes[node.name] = [in_axis]
                elif len(shape) == 2 and 'weights' in node.name:
                    assert (in_axis is not None)
                    assert (in_axis.length == shape[0])
                    out_axis = AxisVar(name=node.name, length=shape[1])
                    name_to_axes[node.name] = [in_axis, out_axis]
                    in_axis = out_axis  # now the output axis becomes input axis for the next layer

    name_to_axes[y_name] = (y_axis, batch_axis)

    return name_to_axes, batch_axis, y_axis


def scan_numerical_axes(graph_def, env):
    """
    Scan the graph to get the numerical axes for each variable.
    Variables are defined and initialized in the next round of graph traversal.

    :param
      - graph_def: a GraphDef object
    :return:
      - names_to_axes: a map from variable name to its axes
      - batch_axis: the batch axis
      - y_axis: axis for labels, not used for inference graph
    """
    name_to_axes = {}
    batch_axis = None
    y_axis = None
    y_name = ""

    for node in graph_def.node:
        inputs = []
        for i, input_name in enumerate([x for x in node.input]):
            inputs.append(input_name)

        op_type = node.op

        with be.bound_environment(env):
            if op_type == 'Placeholder':
                dims = node.attr['shape'].shape
                shape = [d.size for d in dims.dim]

                if batch_axis is None:
                    batch_axis = be.NumericAxis(shape[0])

                if len(shape) == 2:
                    x_axis = be.NumericAxis(shape[1])
                    name_to_axes[node.name] = Axes(x_axis, batch_axis)

                elif len(shape) == 1:
                    name_to_axes[node.name] = (be.NumericAxis(10), batch_axis)
                    y_name = node.name

            elif op_type == 'Variable':
                dims = node.attr['shape'].shape
                shape = [d.size for d in dims.dim]

                if len(shape) == 2:
                    name_to_axes[node.name] = Axes(be.NumericAxis(shape[0]), be.NumericAxis(shape[1]))
                    y_axis = be.NumericAxis(shape[1])
                elif len(shape) == 1:
                    name_to_axes[node.name] = Axes(be.NumericAxis(shape[0]), )
                elif len(shape) == 0:
                    name_to_axes[node.name] = Axes()

            elif op_type == 'Const':
                # in the frozen graph, all variables are converted to constant
                const_tensor = node.attr['value'].tensor
                shape = [d.size for d in const_tensor.tensor_shape.dim]

                if len(shape) == 1 and 'biases' in node.name:
                    name_to_axes[node.name] = Axes(be.NumericAxis(shape[0]), )
                elif len(shape) == 2 and 'weights' in node.name:
                    name_to_axes[node.name] = Axes(be.NumericAxis(shape[0]), be.NumericAxis(shape[1]))

    name_to_axes[y_name] = (y_axis, batch_axis)

    return name_to_axes, batch_axis, y_axis


def _create_nervana_graph(graph_def, env, end_node="", loss_node=""):
    """
    convert the GraphDef object to Neon's graph

    :param
      - graph_def: a (frozen) GraphDef object
    :return:
      - graph: converted graph, including:
       - variables: a map from variable names to variables
       - last_op: the last operator of the graph
       - name_to_op: the map from operation name to its corresponding operator.
                     This structure is similar with TF graph's collection.
    """

    name_to_op = {}  # a map from TF node name to Neon op
    variables = {} # trainable variables

    ignored_nodes = {}

    graph = be.Model()
    graph.x = None
    graph.y = None

    # switched to numerical axes instead of nameable axes
    name_to_axes, batch_axis, y_axis = scan_numerical_axes(graph_def, env)

    print(name_to_axes)

    for node in graph_def.node:
        op_type = node.op

        # skip ignored ops and ops related with serialization.
        if op_type in ignored_ops or 'save' in node.name:
            ignored_nodes[node.name] = True
            continue

        print(node)

        if op_type not in known_ops:
            # TODO: raise unrecognized operator error
            print("unrecognized operator: " + op_type)
            sys.exit()

        inputs = []
        skip_this_node = False
        for i, input_name in enumerate([x for x in node.input]):
            if input_name in ignored_nodes:
                print("inputs contain ignored node: " + input_name + ", skipped")
                skip_this_node = True
                break

            inputs.append(input_name)
            print('inputs[' + str(i) + "]: " + inputs[i])

            if inputs[i] in name_to_op and isinstance(name_to_op[inputs[i]], Tensor):
                print(name_to_op[inputs[i]])

        if skip_this_node:
            ignored_nodes[node.name] = True
            continue

        name_to_op[node.name] = None
        with be.bound_environment(env):
            if op_type in two_inputs_ops:

                if node.name == 'gradients/xentropy_grad/mul':
                    # TODO: remove this hardcoded branch after the ExpandDims op is implemented
                    # use be.Constant(1. / batch_axis.length) as temporal result to replace
                    # the output of ExpandDims (name_to_op[inputs[0]])
                    name_to_op[node.name] = two_inputs_ops[op_type](be.Constant(1. / batch_axis.length),
                                                                    name_to_op[inputs[1]], name=node.name)
                else:
                    name_to_op[node.name] = two_inputs_ops[op_type](name_to_op[inputs[0]],
                                                                    name_to_op[inputs[1]], name=node.name)

            elif op_type in one_inputs_ops:
                name_to_op[node.name] = one_inputs_ops[op_type](name_to_op[inputs[0]])

            elif op_type == 'Relu':
                name_to_op[node.name] = be.maximum(name_to_op[inputs[0]], 0)

            elif op_type == 'Identity':
                name_to_op[node.name] = name_to_op[inputs[0]]

            elif op_type == 'Placeholder':
                dims = node.attr['shape'].shape
                shape = [d.size for d in dims.dim]
                name_to_op[node.name] = be.placeholder(axes=name_to_axes[node.name], name=node.name)
                if len(shape) == 2:
                    graph.x = name_to_op[node.name]
                elif len(shape) == 1:
                    graph.y = name_to_op[node.name]

            elif op_type == 'Const':
                const_tensor = node.attr['value'].tensor
                shape = [d.size for d in const_tensor.tensor_shape.dim]
                np_val = tensor_util.MakeNdarray(const_tensor)

                if node.name in name_to_axes:
                    name_to_op[node.name] = be.NumPyTensor(np_val, axes=name_to_axes[node.name], name=node.name)
                elif len(shape) == 0:
                    name_to_op[node.name] = be.Constant(np_val, name=node.name)
                elif len(shape) == 1:
                    name_to_op[node.name] = be.NumPyTensor(np_val, axes=Axes(be.NumericAxis(shape[0]), ),
                                                           name=node.name)
                elif len(shape) == 2:
                    name_to_op[node.name] = be.NumPyTensor(np_val,
                                                           axes=Axes(be.NumericAxis(shape[0]),
                                                                     be.NumericAxis(shape[1])),
                                                           name=node.name)

            elif op_type == 'Variable':
                name_to_op[node.name] = be.Variable(axes=name_to_axes[node.name], name=node.name)
                variables[node.name] = name_to_op[node.name]

            elif op_type == 'Assign':
                var = name_to_op[inputs[0]]
                init_value = name_to_op[inputs[1]]
                assert (isinstance(var, be.Variable))
                name_to_op[node.name] = be.assign(var, init_value)
                var.initializers.append(name_to_op[node.name])

            elif op_type == 'AssignAdd':
                # TODO: check operations for scala variable
                # Things may broken for other graph in which the scala variable is not
                # named 'global_step'
                if inputs[0] == 'global_step':
                    continue

                var = name_to_op[inputs[0]]
                assert (isinstance(var, be.Variable))
                tensor_to_add = name_to_op[inputs[1]]
                name_to_op[node.name] = be.assign(var, var + tensor_to_add)

            elif op_type == 'Fill':
                # Creates a tensor filled with a scalar value.
                shape_tensor = name_to_op[inputs[0]]
                init_val = name_to_op[inputs[1]]
                assert isinstance(init_val, be.Constant)

                if isinstance(shape_tensor, be.Constant):
                    name_to_op[node.name] = be.Constant(init_val.const, name=node.name)
                else:
                    print(shape_tensor.nptensor)
                    array = np.array(shape_tensor.nptensor)
                    array.fill(init_val.const)
                    print(array)
                    shape = [axis.length for axis in shape_tensor.axes]

                    if len(shape) == 1:
                        name_to_op[node.name] = be.NumPyTensor(array,
                                                               axes=Axes(be.NumericAxis(shape[0])),
                                                               name=node.name)
                    elif len(shape) == 2:
                        name_to_op[node.name] = be.NumPyTensor(array,
                                                               axes=Axes(be.NumericAxis(shape[0]),
                                                                         be.NumericAxis(shape[1])),
                                                               name=node.name)
                    else:
                        assert False

            elif op_type == 'TruncatedNormal' or op_type == 'RandomStandardNormal':
                # TODO: implement tf.truncated_normal and tf.random_normal
                shape_tensor = name_to_op[inputs[0]]  # numpy ndarray
                assert isinstance(shape_tensor, Tensor)
                shape = shape_tensor.nptensor
                print(shape)
                if op_type == 'TruncatedNormal':
                    lower, upper = -2.0, 2.0
                    mu, sigma = 0, 1
                    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                    val = X.rvs(shape)
                elif op_type == "RandomStandardNormal":
                    val = -0.5 + np.random.random_sample(shape).astype(np.float32)

                if len(shape) == 0:
                    name_to_op[node.name] = be.Constant(val, name=node.name)
                elif len(shape) == 1:
                    name_to_op[node.name] = be.NumPyTensor(val, axes=Axes(be.NumericAxis(shape[0]), ), name=node.name)
                elif len(shape) == 2:
                    name_to_op[node.name] = be.NumPyTensor(val, axes=Axes(be.NumericAxis(shape[0]),
                                                                          be.NumericAxis(shape[1]), ), name=node.name)
                else:
                    print("Not supported")
                    assert False

            elif op_type == 'Cast':
                # TODO: need a real cast, currently just skip this op
                dst_type = node.attr['DstT']
                src_type = node.attr['SrcT']
                name_to_op[node.name] = name_to_op[inputs[0]]

            elif op_type == 'SparseSoftmaxCrossEntropyWithLogits':
                # implementation of tf.nn.sparse_softmax_cross_entropy_with_logits
                # check its doc via https://goo.gl/7ytJNB and its C++ implementation via
                # https://goo.gl/z5T2my

                pred = softmax(name_to_op[inputs[0]], Axes(y_axis, ))
                label = name_to_op[inputs[1]]

                name_to_op[node.name] = be.cross_entropy_multi(pred, label, out_axes=(batch_axis,))
                # equivalent: op = -be.sum(safelog(pred) * label * np.float(1. / np.log(2.0)),
                #                             out_axes=(batch_axis,))

                # this op also calculates gradients and saved in the second output
                sum_exp_logits = be.sum(pred, out_axes=(batch_axis,))
                grad = be.divide(pred, sum_exp_logits) - label
                name_to_op[node.name + ":1"] = grad

            elif op_type in reduction_ops:
                input_tensor = name_to_op[inputs[0]]
                assert isinstance(input_tensor, Tensor)
                input_tensor_axes = name_to_op[inputs[0]].axes
                reduction_indices = name_to_op[inputs[1]]
                assert reduction_indices is None or isinstance(reduction_indices, be.NumPyTensor)

                reduction_axes = ()
                if reduction_indices is not None:
                    for i in reduction_indices.nptensor:
                        reduction_axes += Axes(input_tensor_axes[int(i)], )

                name_to_op[node.name] = reduction_ops[op_type](input_tensor,
                                                               reduction_axes=reduction_axes,
                                                               name=node.name)

            elif op_type == 'Prod':
                # TODO: implement tf.reduce_prod and merge with reduction_ops
                reduction_indices = name_to_op[inputs[1]]

                if isinstance(name_to_op[inputs[0]], be.Constant):
                    prod_val = np.prod(name_to_op[inputs[0]].const)
                elif isinstance(name_to_op[inputs[0]], be.NumPyTensor):
                    prod_val = np.prod(name_to_op[inputs[0]].nptensor)
                else:
                    assert False

                name_to_op[node.name] = be.Constant(prod_val, name=node.name)

            elif op_type == 'Shape':
                axes = name_to_op[inputs[0]].axes
                shape = [axis.length for axis in axes]
                print(shape)

                if len(shape) == 0:
                    name_to_op[node.name] = be.Constant(0, name=node.name)
                else:
                    name_to_op[node.name] = be.NumPyTensor(np.array(shape),
                                                           axes=Axes(be.NumericAxis(len(shape)), ),
                                                           name=node.name)

            elif op_type == 'Rank':
                # The rank of a tensor is the number of axis
                shape = name_to_op[inputs[0]].shape
                print(shape)
                name_to_op[node.name] = be.Constant(len(shape), name=node.name)

            elif op_type == 'Size':
                axes = name_to_op[inputs[0]].axes
                shape = [axis.length for axis in axes]
                print(shape)
                name_to_op[node.name] = be.Constant(np.prod(shape), name=node.name)

            elif op_type == 'Range':
                start = name_to_op[inputs[0]]
                limit = name_to_op[inputs[1]]
                delta = name_to_op[inputs[2]]
                nums = np.arange(start.const, limit.const, delta.const).astype(np.float32)
                name_to_op[node.name] = be.NumPyTensor(nums, axes=Axes(be.NumericAxis(len(nums)), ),
                                                       name=node.name)

            elif op_type == 'Mod':
                # TODO: implement tf.mod, currently just skip
                name_to_op[node.name] = name_to_op[inputs[0]]

            elif op_type == 'DynamicStitch':
                # TODO: implemente tf.dynamic_stich, currently just use a constant
                name_to_op[node.name] = be.Constant(1)

            elif op_type == 'Reshape':
                # TODO: implemente tf.reshape
                # Currently it just does nothing but pass the first input without actually reshape
                name_to_op[node.name] = name_to_op[inputs[0]]

            elif op_type == 'Tile':
                # Constructs a tensor by tiling a given tensor. Currently use numpy.tile
                # The first input is the result of tf.reshape, which is currently not available
                # TODO: implement tf.reshape and tf.tile

                input = name_to_op[inputs[0]]
                multiples = name_to_op[inputs[1]]

                # should use the result of multiples as the second arg for np.tile
                # but the value is not available when this graph is constructed.

                array = []
                if isinstance(name_to_op[inputs[0]], be.Constant):
                    array = name_to_op[inputs[0]].const
                elif isinstance(name_to_op[inputs[0]], be.NumPyTensor):
                    array = name_to_op[inputs[0]].nptensor
                val = np.tile(array, batch_axis.length)
                shape = val.shape
                if len(shape) == 1:
                    name_to_op[node.name] = be.NumPyTensor(val, axes=Axes(be.NumericAxis(shape[0]), ), name=node.name)
                else:
                    assert False

            elif op_type == 'ExpandDims':
                # TODO: implement tf.expand_dims
                dim = name_to_op[inputs[1]]
                name_to_op[node.name] = name_to_op[inputs[0]]

            elif op_type == 'BroadcastGradientArgs':
                # implementation of bcast_ops.cc (https://goo.gl/5vx4QN)
                sx = name_to_op[inputs[0]].nptensor
                sy = name_to_op[inputs[1]].nptensor

                grad_x_reduce_ = []
                grad_y_reduce_ = []

                if not np.array_equal(sx, sy):
                    x = sx[::-1]
                    y = sy[::-1]

                    if len(x) > len(y):
                        y = np.pad(y, (0, len(x) - len(y)), 'constant', constant_values=1)
                    else:
                        x = np.pad(x, (0, len(y) - len(x)), 'constant', constant_values=1)

                n = len(x)
                for i in range(n):
                    if not x[i] == y[i]:
                        if x[i] == 1:
                            grad_x_reduce_.append(n - 1 - i)
                        elif y[i] == 1:
                            grad_y_reduce_.append(n - 1 - i)

                print(grad_x_reduce_)
                print(grad_y_reduce_)

                if grad_x_reduce_:
                    val_x = np.array(grad_x_reduce_)
                    name_to_op[node.name] = be.NumPyTensor(val_x,
                                                           axes=Axes(be.NumericAxis(len(grad_x_reduce_)), ),
                                                           name=node.name)

                name_to_op[node.name + ":1"] = None
                if grad_y_reduce_:
                    val_y = np.array(grad_y_reduce_)
                    name_to_op[node.name + ":1"] = \
                        be.NumPyTensor(val_y,
                                       axes=Axes(be.NumericAxis(len(grad_x_reduce_)), ),
                                       name=node.name)

            elif op_type == 'ReluGrad':
                gradient = name_to_op[inputs[0]]
                output = name_to_op[inputs[1]]
                name_to_op[node.name] = gradient * output

            elif op_type == 'ApplyGradientDescent':
                var = name_to_op[inputs[0]]
                lr = name_to_op[inputs[1]]
                grad = name_to_op[inputs[2]]
                updated_var = var - lr * grad
                name_to_op[node.name] = be.assign(var, updated_var)

            elif op_type == 'NoOp':
                # NoOp adds '^' before each original input name
                if node.name == "GradientDescent/update":
                    # gradient descent ops
                    graph.update = be.doall(all=[name_to_op[input[1:]] for input in inputs])
                    name_to_op[node.name] = graph.update

                elif node.name == "init":
                    # variable initialization graph, used only once
                    graph.init = be.doall(all=[name_to_op[input[1:]] for input in inputs[:-1]])
                    name_to_op[node.name] = graph.init

            print(name_to_op[node.name])
            print("---------------------------------------------")

            last_op_name = node.name
            if node.name == end_node:
                print('last_op: ' + last_op_name)
                break

    graph.variables = variables
    graph.last_op = name_to_op[last_op_name]
    graph.name_to_op = name_to_op
    if loss_node in name_to_op:
        graph.loss = name_to_op[loss_node]

    assert (graph.x is not None)

    return graph


def create_nervana_graph(pb_file, env, end_node="", loss_node=""):
    graph_def = tf.GraphDef()
    with open(pb_file, 'rb') as f:
        graph_def.ParseFromString(f.read())

    return _create_nervana_graph(graph_def, env, end_node, loss_node)
