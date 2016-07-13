# ----------------------------------------------------------------------------
# importing TensorFlow GraphDef protobuf and convert to Neon computation graph.
# ----------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import geon.backends.graph.funs as be
from geon.backends.graph.arrayaxes import AxisVar

from tensorflow.python.framework import tensor_util

# known operators that can be processed by Neon graph importer
known_ops = [
  'Add', 'Div', 'MatMul',
  'Identity', 'Relu',
  'Const', 'Variable', 'Placeholder'
]

two_inputs_ops = {
  'Add': be.add,
  'Div': be.divide,
  'MatMul': be.dot,
}

one_inputs_ops = {
  'Relu': be.tanh, # temporarily use tanh as Relu is not implemented
}

def create_neon_graph(graph_def, env):
  '''
  create Neon's transformer graph from a frozen GraphDef protobuf

  :param graph_def: a frozen graph_def protobuf, in which variables are converted to constant
  :return: last operator of the ast graph, all variable names
  '''
  name_to_op = {}
  var_names = []
  graph = be.Model()

  for node in graph_def.node:
    print(node.name)

    if node.op not in known_ops:
      # TODO: raise unrecognized operator error
      print("unrecognized operator: " + op_type)
      continue

    inputs = []
    for i, input_name in enumerate([x for x in node.input]):
      print(node.name + ' | ' + node.op + " <- " + input_name)
      inputs.append(input_name)

    op_type = node.op

    with be.bound_environment(env):
      if op_type in two_inputs_ops:
        op = two_inputs_ops[op_type](name_to_op[inputs[0]], name_to_op[inputs[1]], name=node.name)

      elif op_type in one_inputs_ops:
        op = one_inputs_ops[op_type](name_to_op[inputs[0]])

      elif op_type == 'Identity':
        op = name_to_op[inputs[0]]

      elif op_type == 'Placeholder':
        dims = node.attr['shape'].shape
        shape = [d.size for d in dims.dim]

        if len(shape) == 2:
          batch_axis = AxisVar(name='batch', length=shape[0])
          in_axis = AxisVar(name='x', length=shape[1])
          op = be.placeholder(axes=(in_axis, batch_axis), name='x')
          graph.x = op

        elif len(shape) == 1:

          graph.y = name_to_op[node.name]

      elif op_type == 'Const':
        const_tensor = node.attr['value'].tensor
        np_val = tensor_util.MakeNdarray(const_tensor)

        shape = [d.size for d in const_tensor.tensor_shape.dim]
        print(shape)

        # TODO: op should be initialized with op = be.Constant(np_val, name=node.name)

        if len(shape) == 2:
          if 'weights' in node.name:
            assert (in_axis is not None)
            assert(in_axis.length == shape[0])
            out_axis = AxisVar(name=node.name, length=shape[1])
            op = be.NumPyTensor(np_val, axes=[in_axis, out_axis], name=node.name)

            in_axis = out_axis # now the output axis becomes input axis for the next layer

        elif len(shape) == 1:
          if 'biases' in node.name:
            assert(in_axis is not None)
            assert(in_axis.length == shape[0])
            op = be.NumPyTensor(np_val, axes=[in_axis], name=node.name)

      elif op_type == 'Variable':
        var_names.append(node.name)

      elif op_type == 'sparse_softmax_cross_entropy_with_logits':
        op = be.cross_entropy_multi(name_to_op[inputs[0]], name_to_op[inputs[1]])

      name_to_op[node.name] = op
      last_node = node.name

  graph.var_names = var_names
  graph.last_op = name_to_op[last_node]

  return graph