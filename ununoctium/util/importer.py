# ----------------------------------------------------------------------------
# Utility functions for importing TensorFlow graphs.
# ----------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import geon.backends.graph.funs as be

def create_ast_op(op_type, inputs, name_to_op):
  if op_type == 'Add':
    op = be.add(name_to_op[inputs[0]], name_to_op[inputs[1]])
  elif op_type == 'Const':
    op = be.Constant(0)
  return op

def import_graph_def(graph_def):
  name_to_op = {}

  for node in graph_def.node:
    print(node)

  for node in graph_def.node:
    inputs = []
    for i, input_name in enumerate([x for x in node.input]):
      print(node.name + " <- " + input_name)
      inputs.append(input_name)

    name_to_op[node.name] = create_ast_op(node.op, inputs, name_to_op)
    last_node = node.name

  print(last_node)

  return name_to_op[last_node]