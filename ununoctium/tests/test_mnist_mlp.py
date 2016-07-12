from neon.data import ArrayIterator, load_mnist
from geon.backends.graph.graphneon import *
import geon.backends.graph.analysis as analysis
from geon.backends.graph.environment import Environment

import tensorflow as tf
from util.importer import create_neon_graph

parser = NeonArgparser(__doc__)
args = parser.parse_args()
env = Environment()

(X_train, y_train), (X_test, y_test), nclass = load_mnist(path=args.data_dir)
test_data = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(1, 28, 28))

pb_file = "../../tf_benchmark/mnist/mnist_mlp_graph_froze.pb"

graph_def = tf.GraphDef()
with open(pb_file, 'rb') as f:
    graph_def.ParseFromString(f.read())

ast_graph = create_neon_graph(graph_def, env)

dataflow = analysis.DataFlowGraph([ast_graph.last_op])
dataflow.view()

print(ast_graph.last_op)

with be.bound_environment(env):
     for mb_idx, (xraw, yraw) in enumerate(test_data):
          ast_graph.x.value = xraw.get()

          enp = be.NumPyTransformer(results=[ast_graph.last_op])
          result = enp.evaluate()
          print(result[ast_graph.last_op])

          if mb_idx == 0: break



