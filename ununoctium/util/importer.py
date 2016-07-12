# ----------------------------------------------------------------------------
# importing TensorFlow GraphDef protobuf and convert to Neon computation graph.
# ----------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import geon.backends.graph.funs as be
import geon.backends.graph.axis as ax

from tensorflow.python.framework import tensor_util

# known operators that can be processed by Neon graph backend
known_ops = {
  'Add': be.add,
  'Const': be.Constant,
  'Div': be.divide,
  'MatMul': be.dot,
  'Identity': None,
  'Placeholder': be.placeholder,
  'Relu': be.tanh
}

def create_neon_op(node, name_to_op, env):
  """
  create a corresponding AST node from a NodeDef

  :param node: a NodeDef in the GraphDef
  :param name_to_op: a map from name to existing converted Neon ops.
  :return:
  """

  inputs = []
  for i, input_name in enumerate([x for x in node.input]):
    print(node.name + ' | ' + node.op + " <- " + input_name)
    inputs.append(input_name)

  op_type = node.op

  with be.bound_environment(env):
    if op_type == 'Add':
      op = be.add(name_to_op[inputs[0]], name_to_op[inputs[1]], name=node.name)

    elif op_type == 'Const':
      const_tensor = node.attr['value'].tensor
      np_val = tensor_util.MakeNdarray(const_tensor)

      # TODO: op should be initialized with op = be.Constant(np_val, name=node.name)

      shape = [d.size for d in const_tensor.tensor_shape.dim]
      print(shape)

      if len(shape) == 1:
        if 'biases' in node.name:
          ax.H.length = shape[0]
          op = be.NumPyTensor(np_val, axes=[ax.H], name=node.name)
      elif len(shape) == 2:
        if 'weights' in node.name:
          ax.D.length = shape[0]
          ax.H.length = shape[1]
          op = be.NumPyTensor(np_val, axes=[ax.D, ax.H], name=node.name)

    elif op_type == 'Div':
      op = be.divide(name_to_op[inputs[0]], name_to_op[inputs[1]], name=node.name)

    elif op_type == 'MatMul':
      op = be.dot(name_to_op[inputs[0]], name_to_op[inputs[1]], name=node.name)

    elif op_type == 'Identity':
      op = name_to_op[inputs[0]]

    elif op_type == 'Relu':
      op = be.tanh(name_to_op[inputs[0]])
      # op = be.maximum(name_to_op[inputs[0]],0)

    elif op_type == 'Variable':
      # TODO: create an empty variable node with targeted size
      ax.Y = 1
      op = be.Variable(axes=(ax.Y,), name=node.name)
      # TODO: add the variable to the list of var_to_save

    elif op_type == 'Placeholder':
      dims = node.attr['shape'].shape
      shape = [d.size for d in dims.dim]

      if len(shape) == 2:
        ax.N.length = shape[0]
        ax.D.length = shape[1]
        op = be.placeholder(axes=(ax.D, ax.N), name='x')

    elif op_type == 'sparse_softmax_cross_entropy_with_logits':
      op = be.cross_entropy_multi(name_to_op[inputs[0]], name_to_op[inputs[1]])

    else:
      # TODO: raise unrecognized operator error
      print("unrecognized operator: " + op_type)

  return op


def create_neon_graph(graph_def, env):
  '''
  create Neon's AST graph from a frozen GraphDef protobuf

  :param graph_def: a frozen graph_def protobuf, in which variables are converted to constant
  :return: last operator of the ast graph, all variable names
  '''
  name_to_op = {}
  var_names = []
  graph = be.Model()

  for node in graph_def.node:
    print(node.name)

    if node.op in known_ops:
      name_to_op[node.name] = create_neon_op(node, name_to_op, env)
      last_node = node.name

      if node.op == 'Variable':
        var_names.append(node.name)
      elif node.op == 'Placeholder':
        dims = node.attr['shape'].shape
        shape = [d.size for d in dims.dim]
        if len(shape) == 2:
          graph.x = name_to_op[node.name]
        elif len(shape) == 1:
          graph.y = name_to_op[node.name]

  graph.var_names = var_names
  graph.last_op = name_to_op[last_node]

  return graph