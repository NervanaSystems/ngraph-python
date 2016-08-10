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

To Run:

    python train_mnist_mlp.py --loss_node="xentropy_mean"

"""

from __future__ import print_function
from neon.data import MNIST
from neon.util.argparser import NeonArgparser

import geon as be
import geon.util.analysis as analysis
from geon.backends.graph.environment import Environment

from util.importer import create_nervana_graph
import numpy as np
import sys

parser = NeonArgparser(__doc__)
parser.set_defaults(backend='dataloader')
parser.add_argument('--pb_file', type=str, default="mnist/graph.pb",
                    help='GraphDef protobuf')
parser.add_argument('--end_node', type=str, default="",
                    help='the last node to execute, mainly used for debugging')
parser.add_argument('--loss_node', type=str, default="",
                    help='the node name to calculate the loss values during training')

args = parser.parse_args()

env = Environment()

# TODO: load meta info from TF's MetaGraph, including details about dataset, training epochs and etc

epochs = 5

mnist_data = MNIST(path=args.data_dir).gen_iterators()
train_data = mnist_data['train']
test_data = mnist_data['valid']


nervana_graph = create_nervana_graph(args.pb_file, env, args.end_node, args.loss_node)

if args.end_node == "":
    transformer = be.NumPyTransformer(results=[nervana_graph.update])
    transformer.dataflow.view()

else:
    transformer = be.NumPyTransformer(results=[nervana_graph.last_op])
    transformer.dataflow.view()

def eval_test(test_data, graph, last_op_name):
    # TODO: pass the inference graph instead of manually specify the last node for inference.
    """
    :param test_data: test input
    :param graph: the whole computation graph
    :param last_op_name: the last op for inference
    :return: error rate (1 - accuracy) on test_data
    """
    with be.bound_environment(env):
        test_error = 0
        n_bs = 0
        enp = be.NumPyTransformer(results=[graph.name_to_op[last_op_name]])
        for mb_idx, (xraw, yraw) in enumerate(test_data):
            graph.x.value = xraw
            result = enp.evaluate()[graph.name_to_op[last_op_name]]
            pred = np.argmax(result, axis=1)
            gt = np.argmax(yraw, axis=0)
            test_error += np.sum(np.not_equal(pred, gt))
            n_bs += 1

        bsz = result.shape[0]
        return test_error / float(bsz) / n_bs * 100

with be.bound_environment(env):
    # initialize all variables with the init op
    if args.end_node == "":
        enp = be.NumPyTransformer(results=[nervana_graph.init])
        enp.evaluate()

    for epoch in range(epochs):
        print("===============================")
        print("epoch: " + str(epoch))

        avg_loss = 0
        for mb_idx, (xraw, yraw) in enumerate(train_data):
            nervana_graph.x.value = xraw
            nervana_graph.y.value = yraw

            if args.end_node == "":
                enp = be.NumPyTransformer(results=[nervana_graph.loss, nervana_graph.update])
                result = enp.evaluate()
                avg_loss += result[nervana_graph.loss]
            else:
                enp = be.NumPyTransformer(results=[nervana_graph.name_to_op[args.end_node]])
                result = enp.evaluate()[nervana_graph.name_to_op[args.end_node]]

                print("-------------------------------")
                print("minibatch: " + str(mb_idx))
                print("-------------------------------")
                print("the last op: ")
                print(nervana_graph.last_op)
                print("result of the last op: ")
                print(result)
                print("shape of the result: ")
                print(result.shape)
                sys.exit()

        avg_loss /= mb_idx
        test_error = eval_test(test_data, nervana_graph, "softmax_linear/add")
        print("train_loss: %.2f test_error: %.2f" % (avg_loss, test_error))