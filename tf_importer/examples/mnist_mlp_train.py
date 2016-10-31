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
TODO: load meta info from TF's MetaGraph, including details about dataset,
      training epochs and etc
TODO: pass the inference computation graph only without provide the last node
      for inference (in eval_test())
TODO: infer_node: should determine automatically or receive as arg parameter
"""

from __future__ import print_function, division

import ngraph as ng
import numpy as np
from neon.data import MNIST
from neon.util.argparser import NeonArgparser
from tf_importer.tf_importer.importer import TFImporter

parser = NeonArgparser(__doc__)
parser.add_argument('--pb_file', type=str, default="graph.pb",
                    help='GraphDef protobuf')
parser.add_argument('--end_node', type=str, default="",
                    help='the last node to execute, mainly used for debugging')
parser.add_argument('--loss_node', type=str, default="xentropy_mean",
                    help='the node name to calculate the loss values during training')
parser.add_argument('--infer_node', type=str, default="softmax_linear/add",
                    help='the node name to produce the prediction probability')

args = parser.parse_args()


def eval_test(test_data, tf_importer, inference_comp):
    """
    Evaluate on test_data

    Arguments:
        test_data: test input data
        tf_importer (TFImporter): tensorflow importer
        inference_comp (Computation): computation for inference

    Returns:
        float: error rate
    """
    test_error = 0
    n_sample = 0
    for mb_idx, (xraw, yraw) in enumerate(test_data):
        tf_importer.x.value[:] = xraw
        result = inference_comp()
        pred = np.argmax(result, axis=1)
        gt = np.argmax(yraw, axis=0)
        test_error += np.sum(np.not_equal(pred, gt))
        n_sample += pred.shape[0]

    return test_error / float(n_sample)


def train_mnist_mlp():
    """
    Runs mnist mlp train example

    Returns:
        None
    """

    # dataset
    mnist_data = MNIST(path=args.data_dir).gen_iterators()
    train_data = mnist_data['train']
    test_data = mnist_data['valid']

    # tf_importer
    tf_importer = TFImporter(args.pb_file,
                             end_node_name=args.end_node,
                             loss_node_name=args.loss_node)

    # init computation
    init_comp = None
    transformer = ng.NumPyTransformer()
    if tf_importer.init_op is not None:
        init_comp = transformer.computation([tf_importer.init_op])

    # inference computation
    pred_op = tf_importer.name_to_op[args.infer_node]
    predict_comp = transformer.computation(pred_op)

    # update computation
    update_comp = transformer.computation([tf_importer.loss_op,
                                           tf_importer.update_op],
                                          tf_importer.x, tf_importer.y)

    # initialize transformer
    transformer.allocate()
    transformer.initialize()

    # visualize
    # transformer.dataflow.view()

    # run init computation once
    init_comp()

    for epoch_index in range(20):
        print("===============================")
        print("epoch: %s" % epoch_index)

        total_loss = 0.
        for mb_idx, (x_raw, y_raw) in enumerate(train_data):
            total_loss += update_comp(x_raw, y_raw)[0]
        average_loss = total_loss / float(mb_idx)

        test_error = eval_test(test_data, tf_importer, predict_comp)
        print("train_loss: %.2f test_error: %.2f" % (average_loss,
                                                     test_error * 100.))


if __name__ == '__main__':
    train_mnist_mlp()
