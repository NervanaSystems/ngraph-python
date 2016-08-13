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
parser.add_argument('--loss_node', type=str, default="xentropy_mean",
                    help='the node name to calculate the loss values during training')
parser.add_argument('--infer_node', type=str, default="softmax_linear/add",
                    help='the node name to produce the prediction probability')

args = parser.parse_args()

env = Environment()

# TODO: load meta info from TF's MetaGraph, including details about dataset, training epochs and etc

epochs = 1

mnist_data = MNIST(path=args.data_dir).gen_iterators()
train_data = mnist_data['train']
test_data = mnist_data['valid']

nervana_graph = create_nervana_graph(args.pb_file, env, args.end_node, args.loss_node)

init_comp = None
trans = be.NumPyTransformer()
if nervana_graph.init is not None:
    init_comp = trans.computation([nervana_graph.init])

inference_comp = None
if args.infer_node in nervana_graph.name_to_op:
    # TODO: should determine automatically or receive as arg parameter
    pred_op = nervana_graph.name_to_op[args.infer_node]
    inference_comp = trans.computation([pred_op])

debug_comp = None
debug_op = None
if args.end_node in nervana_graph.name_to_op:
    debug_op = nervana_graph.name_to_op[args.end_node]
    debug_comp = trans.computation([debug_op])

update_comp = None
if nervana_graph.loss is not None and nervana_graph.update is not None:
    if debug_op is not None:
        update_comp = trans.computation([nervana_graph.loss, nervana_graph.update, debug_op])
    else:
        update_comp = trans.computation([nervana_graph.loss, nervana_graph.update])

trans.finalize()
trans.dataflow.view()

def eval_test(test_data, graph, infernce_comp, pred_op):
    # TODO: pass the inference computation graph only without provide the last node for inference.
    """
    :param test_data: test input
    :param inference_comp: the computation graph for inference
    :param pred_op: the last op for inference
    :param inference_comp: the transformer.computation
    :return: error rate (1 - accuracy) on test_data
    """
    with be.bound_environment(env):
        test_error = 0
        n_sample = 0
        for mb_idx, (xraw, yraw) in enumerate(test_data):
            graph.x.value = xraw
            result = infernce_comp.evaluate()[pred_op]
            pred = np.argmax(result, axis=1)
            gt = np.argmax(yraw, axis=0)
            test_error += np.sum(np.not_equal(pred, gt))
            n_sample += pred.shape[0]

        return test_error / float(n_sample) * 100

with be.bound_environment(env):
    # initialize all variables with the init op
    if init_comp is None:
        print("Initialization is not completed.")
        sys.exit()

    init_comp.evaluate()

    for epoch in range(epochs):
        print("===============================")
        print("epoch: " + str(epoch))

        avg_loss = 0
        for mb_idx, (xraw, yraw) in enumerate(train_data):
            nervana_graph.x.value = xraw
            nervana_graph.y.value = yraw

            result = update_comp.evaluate()
            avg_loss += result[nervana_graph.loss]

            if mb_idx % 1000 == 0:
                print("epoch: %d minibatch: %d" % (epoch, mb_idx))

                print("the last op: ")
                print(debug_op)
                print("result of the last op: ")
                print(result[debug_op])
                print("shape of the result: ")
                print(result[debug_op].shape)

                # print out variables
                for v in nervana_graph.variables:
                    print(v)
                    val = nervana_graph.variables[v].tensor_description(trans).value
                    print(val)
                    if val is not None and np.isnan(val).any(): sys.exit()

                print("-------------------------------")

        avg_loss /= mb_idx
        test_error = eval_test(test_data, nervana_graph, inference_comp, pred_op)
        print("train_loss: %.2f test_error: %.2f" % (avg_loss, test_error))