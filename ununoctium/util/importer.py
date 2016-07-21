# ----------------------------------------------------------------------------
# importing TensorFlow GraphDef from a protobuf file and convert it to Neon's computation graph.
# ----------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function
from builtins import str

import geon.backends.graph.funs as be
from geon.backends.graph.arrayaxes import AxisVar
from geon.backends.graph.transform import Tensor, Constant

from tensorflow.python.framework import tensor_util
import numpy as np

# known operators that can be processed by Neon graph importer
known_ops = [
  'Add', 'Div', 'MatMul', 'Maximum', 'Mean', 'Mul', 'Mod',
  'Identity', 'Relu',
  'Const', 'Variable', 'Placeholder', 'Range',
  'Assign', 'Cast',
  'SparseSoftmaxCrossEntropyWithLogits',
  'Shape', 'Rank', 'Size',
  'Prod',
  'TruncatedNormal',
  'Fill', 'Tile',
  'DynamicStitch', 'Reshape',
]

two_inputs_ops = {
  'Add': be.add,
  'Div': be.divide,
  'MatMul': be.dot,
  'Maximum': be.maximum,
  'Mul': be.multiply,
}

one_inputs_ops = {
  'Relu': be.tanh, # temporarily use tanh as Relu is not implemented
}


ignore_ops = {
  'ScalarSummary', 'ZerosLike',
}

def scan_variables(graph_def, env):
  """
  Scan the graph to get the info of axis/initialization for variables.
  Variables are defined and initialized in the next round of graph traversal.

  """
  name_to_axes = {}
  batch_axis = None

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
          batch_axis = AxisVar(name='batch', length=shape[0])

        if len(shape) == 2:
          in_axis = AxisVar(name=str(shape[1]), length=shape[1])
          name_to_axes[node.name] = (in_axis, batch_axis)
        elif len(shape) == 1:
          y_axis = AxisVar(name='y', length=10)
          name_to_axes[node.name] = (y_axis, batch_axis)

      elif op_type == 'Variable':
        dims = node.attr['shape'].shape
        shape = [d.size for d in dims.dim]

        if len(shape) == 2:
          if 'weights' in node.name:
            assert (in_axis is not None)
            assert (in_axis.length == shape[0])
            out_axis = AxisVar(name=str(shape[1]), length=shape[1])
            name_to_axes[node.name] = (in_axis, out_axis)
            in_axis = out_axis  # now the output axis becomes input axis for the next layer

        elif len(shape) == 1:
          if 'biases' in node.name:
            assert (in_axis is not None)
            assert (in_axis.length == shape[0])
            name_to_axes[node.name] = (in_axis,)

        elif len(shape) == 0:
          name_to_axes[node.name] = (AxisVar(name=node.name, length=1),)

  return name_to_axes, batch_axis

def create_neon_graph(graph_def, env, end_node=None):
  '''
  create Neon's transformer graph from a frozen GraphDef protobuf

  :param graph_def: a frozen graph_def protobuf, in which variables are converted to constant
  :return: last operator of the ast graph, all variable names
  '''
  name_to_op = {}
  var_names = []
  graph = be.Model()

  name_to_axes, batch_axis = scan_variables(graph_def, env)

  for node in graph_def.node:
    op_type = node.op

    if op_type in ignore_ops:
      continue

    if op_type not in known_ops:
      # TODO: raise unrecognized operator error
      print("unrecognized operator: " + op_type)
      break

    print(node)

    inputs = []
    for i, input_name in enumerate([x for x in node.input]):
      inputs.append(input_name)
      print('input[' + str(i) + "]:")
      print(name_to_op[inputs[i]])

    with be.bound_environment(env):
      if op_type in two_inputs_ops:
        assert isinstance(name_to_op[inputs[0]], Tensor) and isinstance(name_to_op[inputs[1]], Tensor)
        if isinstance(name_to_op[inputs[0]], Constant) \
                and isinstance(name_to_op[inputs[1]], Constant) \
                and op_type == 'Mul':
          result = np.multiply(name_to_op[inputs[0]].const, name_to_op[inputs[1]].const)
          print(result.dtype)
          op = be.Constant(result)
        else:
          op = two_inputs_ops[op_type](name_to_op[inputs[0]], name_to_op[inputs[1]], name=node.name)

      elif op_type in one_inputs_ops:
        op = one_inputs_ops[op_type](name_to_op[inputs[0]])
      elif op_type == 'Identity':
        op = name_to_op[inputs[0]]
      elif op_type == 'Placeholder':
        dims = node.attr['shape'].shape
        shape = [d.size for d in dims.dim]
        op = be.placeholder(axes=name_to_axes[node.name], name=node.name)
        if len(shape) == 2:
          graph.x = op
        elif len(shape) == 1:
          graph.y = op
      elif op_type == 'Const':
        const_tensor = node.attr['value'].tensor
        shape = [d.size for d in const_tensor.tensor_shape.dim]
        np_val = tensor_util.MakeNdarray(const_tensor)

        if 'weights' in node.name:
          assert(len(shape) == 2)
          assert(in_axis is not None)
          assert(in_axis.length == shape[0])
          out_axis = AxisVar(name=node.name, length=shape[1])
          op = be.NumPyTensor(np_val, axes=[in_axis, out_axis], name=node.name)
          in_axis = out_axis # now the output axis becomes input axis for the next layer
        elif 'biases' in node.name:
          assert(len(shape) == 1)
          assert(in_axis is not None)
          assert(in_axis.length == shape[0])
          op = be.NumPyTensor(np_val, axes=[in_axis], name=node.name)
        else:
          op = be.Constant(np_val, name=node.name)

      elif op_type == 'Variable':
        op = be.Variable(axes=name_to_axes[node.name], name=node.name)

      elif op_type == 'Assign':
        var = name_to_op[inputs[0]]
        init_value = name_to_op[inputs[1]]
        assert(isinstance(var, be.Variable))
        assert (isinstance(init_value, Tensor))
        var.initializers.append(init_value)
        # op = be.assign(var, init_value)
      elif op_type == 'Fill':
        shape = name_to_op[inputs[0]]
        init_val = name_to_op[inputs[1]]
        assert isinstance(shape, Tensor)
        assert isinstance(init_val, Constant)
        array = np.array(shape.const)
        if not array:
          op = be.Constant(init_val.const)
        else:
          array.fill(init_val.const)
          op = be.Constant(array)
      elif op_type == 'TruncatedNormal':
        #TODO:
        shape = name_to_op[inputs[0]] # numpy ndarray
        assert isinstance(shape, Tensor)
        shape = tuple(shape.const)
        val = np.random.random_sample(shape).astype(np.float32)
        op = be.Constant(val, name=node.name)
      elif op_type == 'Cast':
        dst_type = node.attr['DstT']
        src_type = node.attr['SrcT']
        #TODO: currently just use the original format, need a real cast
        op = name_to_op[inputs[0]]

      elif op_type == 'SparseSoftmaxCrossEntropyWithLogits':
        logscale = -np.float(1. / np.log(2.0))
        op = be.sum(be.safelog(name_to_op[inputs[0]]) * name_to_op[inputs[1]], out_axes=(batch_axis,)) * logscale

      elif op_type == 'Mean':
        # TODO: use the attribute of kee_dims
        keep_dims = node.attr['keep_dims']
        op = be.mean(name_to_op[inputs[0]], name=node.name)
      elif op_type == 'Shape':
        assert (isinstance(name_to_op[inputs[0]], Tensor))
        shape = name_to_op[inputs[0]].tensor_axes_info.tensor_description.shape
        op = be.Constant(shape, name=node.name)
      elif op_type == 'Rank':
        assert(isinstance(name_to_op[inputs[0]], Tensor))
        shape = name_to_op[inputs[0]].tensor_axes_info.tensor_description.shape
        rank = len(shape)
        op = be.Constant(rank, name=node.name)
      elif op_type == 'Size':
        assert(isinstance(name_to_op[inputs[0]], Tensor))
        shape = name_to_op[inputs[0]].tensor_axes_info.tensor_description.shape
        op = be.Constant(np.prod(shape), name=node.name)
      elif op_type == 'Range':
        assert(len(inputs) == 3)
        start = name_to_op[inputs[0]]
        limit = name_to_op[inputs[1]]
        delta = name_to_op[inputs[2]]
        print(start + ", " + limit + " " + delta)
        op = np.arange(start.const, limit.const, delta.const)
        op = be.Constant(op, name=node.name)
        # range = np.arange(start, limit, delta)
        # op = be.NumPyTensor(range, axes=AxisVar(name=node.name, length=len(shape)), name=node.name)

      elif op_type == 'Prod':
        #TODO: use be.reduce_prod
        keep_dims = node.attr['keep_dims']
        op = np.prod(name_to_op[inputs[0]])
        op = be.Constant(op, name=node.name)

      elif op_type == 'Mod':
        #TODO: implement tf.mod
        assert (isinstance(name_to_op[inputs[0]], Tensor))
        assert (isinstance(name_to_op[inputs[1]], Tensor))
        op = name_to_op[inputs[0]]

      elif op_type == 'DynamicStitch':
        #TODO: implemente tf.dynamic_stich
        op = be.Constant(1)
      elif op_type == 'Reshape':
        # TODO: implemente tf.reshape
        print('inputs[0]:' + inputs[0])
        print(name_to_op[inputs[0]])
        print('inputs[1]:' + inputs[1])
        print(name_to_op[inputs[1]])
        op = name_to_op[inputs[0]]
      elif op_type == 'Tile':
        # TODO: implement tf.tile
        # be.tile is not available, we use hard coded number instead
        val = np.tile(name_to_op[inputs[0]].const, 128)
        op = be.Constant(val)

      print("output:")
      print(op)

      print("---------------------------------------------")

      name_to_op[node.name] = op
      last_op_name = node.name

      if node.name == end_node:
        print('last_op: ' + last_op_name)
        break

  graph.var_names = var_names
  graph.last_op = name_to_op[last_op_name]

  return graph
