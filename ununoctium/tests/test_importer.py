# ----------------------------------------------------------------------------
# load TensorFlow's computation graph in protobuf and convert it to Neon's AST graph
# ----------------------------------------------------------------------------

# note that the tensorflow package imported here is the local graphiti version.

# in case the .pb file does not exist, you might need to first execute the TensorFlow script first
# cd ../../tf_benchmark/
# python create_sample_graph.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from util.importer import import_graph_def
import geon.backends.graph.analysis as analysis

def load_tf_graph(pb_file):
  print("loading graph")
  graph_def = tf.GraphDef()

  with open(pb_file, 'rb') as f:
    graph_def.ParseFromString(f.read()) # read serialized binary file only
    imported_graph = import_graph_def(graph_def)

    dataflow = analysis.DataFlowGraph([imported_graph])
    dataflow.view()

load_tf_graph("../../tf_benchmark/sample/sample_graph.pb")
