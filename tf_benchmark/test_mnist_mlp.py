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
from __future__ import print_function
from neon.data import ArrayIterator, load_mnist
from geon.backends.graph.graphneon import *  # noqa
import geon.backends.graph.analysis as analysis
from geon.backends.graph.environment import Environment

import tensorflow as tf
from util.importer import create_neon_graph

parser = NeonArgparser(__doc__)
parser.set_defaults(backend='dataloader')
parser.add_argument('--pb_file', type=str, default="mnist/mnist_mlp_graph.pb",
                    help='GraphDef protobuf')
parser.add_argument('--end_node', type=str, default=None,
                    help='the last node to execute')

args = parser.parse_args()

env = Environment()

(X_train, y_train), (X_test, y_test), nclass = load_mnist(path=args.data_dir)
test_data = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(1, 28, 28))

graph_def = tf.GraphDef()
with open(args.pb_file, 'rb') as f:
    graph_def.ParseFromString(f.read())

ast_graph = create_neon_graph(graph_def, env, args.end_node)

dataflow = analysis.DataFlowGraph([ast_graph.last_op])
dataflow.view()

print(ast_graph.last_op)

with be.bound_environment(env):
    for mb_idx, (xraw, yraw) in enumerate(test_data):
        ast_graph.x.value = xraw
        ast_graph.y.value = yraw

        enp = be.NumPyTransformer(results=[ast_graph.last_op])
        result = enp.evaluate()[ast_graph.last_op]
        print(result)

        if mb_idx == 0:
            break
