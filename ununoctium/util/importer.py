# ----------------------------------------------------------------------------
# importing TensorFlow GraphDef protobuf and convert to Neon computation graph.
# ----------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function
from builtins import str

from neon.initializers import Uniform
import geon.backends.graph.funs as be
from geon.backends.graph.arrayaxes import AxisVar

from tensorflow.python.framework import tensor_util
import numpy as np

# known operators that can be processed by Neon graph importer
known_ops = [
  'Add', 'Div', 'MatMul', 'Maximum', 'Mean',
  'Identity', 'Relu',
  'Const', 'Variable', 'Placeholder', 'Range',
  'Assign', 'Cast',
  'SparseSoftmaxCrossEntropyWithLogits',
  'Shape', 'Rank', 'Size',
  'Prod',
]

two_inputs_ops = {
  'Add': be.add,
  'Div': be.divide,
  'MatMul': be.dot,
  'Maximum': be.maximum,
  # 'Mul': np.multiply,
}

one_inputs_ops = {
  'Relu': be.tanh, # temporarily use tanh as Relu is not implemented
  'TruncatedNormal': np.random #temporarily use tanh as TruncatedNormal is not implemented
}

def scan_variables(graph_def, env):
  """
  Scan the graph to get the knowledge of axis dependence for variables.
  Variables are defined and initialized in the next round of graph traversal.

  """
  var_to_init = {}
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

      elif op_type == 'Assign':
        # init_func = name_to_op[inputs[1]]
        init_func = Uniform(-.001, .001)
        var_to_init[inputs[0]] = init_func

  return name_to_axes, var_to_init, batch_axis

def create_neon_graph(graph_def, env):
  '''
  create Neon's transformer graph from a frozen GraphDef protobuf

  :param graph_def: a frozen graph_def protobuf, in which variables are converted to constant
  :return: last operator of the ast graph, all variable names
  '''
  name_to_op = {}
  var_names = []
  graph = be.Model()

  name_to_axes, var_to_init, batch_axis = scan_variables(graph_def, env)

  for node in graph_def.node:
    if node.name == 'ScalarSummary/tags':
      break

    if node.op not in known_ops:
      # TODO: raise unrecognized operator error
      print("unrecognized operator: " + op_type)
      continue

    inputs = []
    for i, input_name in enumerate([x for x in node.input]):
      inputs.append(input_name)

    op_type = node.op

    with be.bound_environment(env):
      if op_type in two_inputs_ops:
        if (inputs[0] not in name_to_op or inputs[1] not in name_to_op):
          continue
        op = two_inputs_ops[op_type](name_to_op[inputs[0]], name_to_op[inputs[1]], name=node.name)

      elif op_type in one_inputs_ops:
        op = one_inputs_ops[op_type](name_to_op[inputs[0]])

      elif op_type == 'Identity':
        op = name_to_op[inputs[0]]

      elif op_type == 'Mul':
        op = np.multiply(name_to_op[inputs[0]], name_to_op[inputs[1]])

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

        # TODO: op should be initialized with op = be.Constant(np_val, name=node.name)

        if len(shape) == 2:
          if 'weights' in node.name:
            assert (in_axis is not None)
            assert(in_axis.length == shape[0])
            out_axis = AxisVar(name=node.name, length=shape[1])
            op = be.NumPyTensor(np_val, axes=[in_axis, out_axis], name=node.name)
            in_axis = out_axis # now the output axis becomes input axis for the next layer
          else:
            op = np_val
        elif len(shape) == 1:
          if 'biases' in node.name:
            assert(in_axis is not None)
            assert(in_axis.length == shape[0])
            op = be.NumPyTensor(np_val, axes=[in_axis], name=node.name)
          else:
            op = np_val

      elif op_type == 'Variable':
        op = be.Variable(axes=name_to_axes[node.name], init=Uniform(-.001, .001), name=node.name)

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
        op = be.mean(name_to_op[inputs[0]])
      elif op_type == 'Shape':
        op = name_to_op[inputs[0]].shape
      elif op_type == 'Rank':
        op = len(name_to_op[inputs[0]].shape)
      elif op_type == 'Size':
        op = name_to_op[inputs[0]].size
      elif op_type == 'Range':
        assert(len(inputs) == 3)
        start = inputs[0]
        limit = inputs[1]
        delta = inputs[2]
        op = np.range(start, limit, delta)
      elif op_type == 'Prod':
        #TODO: use be.reduce_prod
        keep_dims = inputs[1]
        op = np.prod(inputs[0])

      name_to_op[node.name] = op
      last_node = node.name

  graph.var_names = var_names
  graph.last_op = name_to_op[last_node]


  return graph
