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
This script illustrates how to convert a pre-trained TF model to Neon and perform
inference with the model on new data.

"""

from __future__ import print_function
from neon.data import ArrayIterator, load_mnist
from neon.util.argparser import NeonArgparser
import geon.backends.graph.funs as be
import geon.backends.graph.analysis as analysis
from geon.backends.graph.environment import Environment

import tensorflow as tf
from util.importer import create_nervana_graph
import numpy as np

parser = NeonArgparser(__doc__)
parser.set_defaults(backend='dataloader')
parser.add_argument('--pb_file', type=str, default="mnist/graph_froze.pb",
                    help='GraphDef protobuf')

args = parser.parse_args()

env = Environment()

# TODO: load meta info from TF's MetaGraph, including details about dataset, training epochs and etc

(X_train, y_train), (X_test, y_test), nclass = load_mnist(path=args.data_dir)
test_data = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(1, 28, 28))

graph_def = tf.GraphDef()
with open(args.pb_file, 'rb') as f:
    graph_def.ParseFromString(f.read())

nervana_graph = create_nervana_graph(graph_def, env, args.end_node)

dataflow = analysis.DataFlowGraph([nervana_graph.last_op])
dataflow.view()

def eval_test(test_data, graph):
    with be.bound_environment(env):
        test_error = 0
        n_bs = 0
        enp = be.NumPyTransformer(results=[graph.last_op])
        for mb_idx, (xraw, yraw) in enumerate(test_data):
            graph.x.value = xraw
            result = enp.evaluate()[graph.last_op]

            print("-------------------------------")
            print("minibatch: " + str(mb_idx))
            print("-------------------------------")
            print("prediction result: ")
            print(result)
            print("shape of the prediction: ")
            print(result.shape)

            pred = np.argmax(result, axis=1)
            gt = np.argmax(yraw, axis=0)
            print(pred)
            print(gt)
            test_error += np.sum(np.not_equal(pred, gt))
            n_bs += 1

        bsz = result.shape[0]
        return test_error / float(bsz) / n_bs * 100

test_error = eval_test(test_data, nervana_graph)
print("-------------------------------")
print("test error: %2.2f %%" % test_error)
