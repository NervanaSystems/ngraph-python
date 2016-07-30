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
This script illustrates how to import a model that was defined by a TF script and
train the model from scratch with Neon following the original script's specification.

"""

from __future__ import print_function
from neon.data import ArrayIterator, load_mnist
from geon.backends.graph.graphneon import *  # noqa
import geon.backends.graph.analysis as analysis
from geon.backends.graph.environment import Environment

import tensorflow as tf
from util.importer import create_nervana_graph

parser = NeonArgparser(__doc__)
parser.set_defaults(backend='dataloader')
parser.add_argument('--pb_file', type=str, default="mnist/mnist_mlp_graph.pb",
                    help='GraphDef protobuf')
parser.add_argument('--end_node', type=str, default="",
                    help='the last node to execute')

args = parser.parse_args()

env = Environment()

# TODO: load meta info from TF's MetaGraph, including details about dataset, training epochs and etc

epochs = 1

(X_train, y_train), (X_test, y_test), nclass = load_mnist(path=args.data_dir)
train_data = ArrayIterator(X_train, y_train, nclass=nclass, lshape=(1, 28, 28))
test_data = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(1, 28, 28))

graph_def = tf.GraphDef()
with open(args.pb_file, 'rb') as f:
    graph_def.ParseFromString(f.read())

nervana_graph = create_nervana_graph(graph_def, env, args.end_node)
assert (nervana_graph.x is not None)
assert (nervana_graph.y is not None)

if args.end_node == "":
    dataflow = analysis.DataFlowGraph([nervana_graph.update])
else:
    dataflow = analysis.DataFlowGraph([nervana_graph.last_op])

dataflow.view()

with be.bound_environment(env):
    # initialize all variables with the init op
    enp = be.NumPyTransformer(results=[nervana_graph.init])

    for epoch in range(epochs):
        print("===============================")
        print("epoch: " + str(epoch))
        print("===============================")

        for mb_idx, (xraw, yraw) in enumerate(train_data):
            nervana_graph.x.value = xraw
            nervana_graph.y.value = yraw

            if args.end_node == "":
                enp = be.NumPyTransformer(results=[nervana_graph.update])
                result = enp.evaluate()[nervana_graph.update]
            else:
                enp = be.NumPyTransformer(results=[nervana_graph.last_op])
                result = enp.evaluate()[nervana_graph.last_op]

            print("-------------------------------")
            print("minibatch: " + str(mb_idx))
            print("-------------------------------")
            print("result of the last op: ")
            print(result)
            print("shape of the result: ")
            print(result.shape)
            print("softmax_linear/biases:")
            print(nervana_graph.name_to_op["softmax_linear/biases"].value)

            # execute one minibatch for test only
            if mb_idx == 10:
                break
