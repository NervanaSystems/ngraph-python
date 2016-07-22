from __future__ import print_function
from neon.data import ArrayIterator, load_mnist
from geon.backends.graph.graphneon import *  # noqa
import geon.backends.graph.analysis as analysis
from geon.backends.graph.environment import Environment

import tensorflow as tf
from util.importer import create_neon_graph

parser = NeonArgparser(__doc__)
parser.set_defaults(backend='dataloader')
args = parser.parse_args()

env = Environment()

(X_train, y_train), (X_test, y_test), nclass = load_mnist(path=args.data_dir)
test_data = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(1, 28, 28))

# Inference Graph
pb_file = "../../tf_benchmark/mnist/mnist_mlp_graph_froze.pb"


# Training graph
pb_file = "../../tf_benchmark/mnist/mnist_mlp_graph.pb"
end_node = 'ScalarSummary/tags'

graph_def = tf.GraphDef()
with open(pb_file, 'rb') as f:
    graph_def.ParseFromString(f.read())

ast_graph = create_neon_graph(graph_def, env, end_node)

dataflow = analysis.DataFlowGraph([ast_graph.last_op])
dataflow.view()

print(ast_graph.last_op)

with be.bound_environment(env):
    for mb_idx, (xraw, yraw) in enumerate(test_data):
        # ast_graph.x.value = xraw.get()
        ast_graph.x.value = xraw
        ast_graph.y.value = yraw

        enp = be.NumPyTransformer(results=[ast_graph.last_op])
        result = enp.evaluate()[ast_graph.last_op]
        print(result)

        if mb_idx == 0:
            break
