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
from neon.data import MNIST
from neon.util.argparser import NeonArgparser

import geon as be
from geon.backends.graph.environment import Environment

from util.importer import create_nervana_graph
import numpy as np

parser = NeonArgparser(__doc__)
parser.set_defaults(backend='dataloader')
parser.add_argument('--pb_file', type=str, default="mnist/graph_froze.pb",
                    help='GraphDef protobuf')

args = parser.parse_args()

env = Environment()

# TODO: load meta info from TF's MetaGraph, including details about dataset, training epochs and etc

mnist_data = MNIST(path=args.data_dir).gen_iterators()
test_data = mnist_data['valid']

nervana_graph = create_nervana_graph(args.pb_file, env)  # trans = be.NumPyTransformer()

with be.bound_environment(env):
    trans = be.NumPyTransformer()
    infer_comp = trans.computation([nervana_graph.last_op])
    trans.finalize()
    trans.dataflow.view()

    test_error = 0
    n_sample = 0

    y_raw_1 = None
    for mb_idx, (xraw, yraw) in enumerate(test_data):
        nervana_graph.x.value = xraw
        yraw1 = yraw

        result = infer_comp.evaluate()[nervana_graph.last_op]

        print("minibatch: " + str(mb_idx))
        print("prediction result: ")
        print(result)
        print("shape of the prediction: ")
        print(result.shape)

        pred = np.argmax(result, axis=1)
        gt = np.argmax(yraw1, axis=0)
        mb_test_error = np.sum(np.not_equal(pred, gt))

        print("test error: %2.2f %%" % (mb_test_error / float(xraw.shape[1]) * 100))
        test_error += mb_test_error
        n_sample += xraw.shape[1]

        print("-------------------------------")

    test_error = test_error / float(n_sample) * 100
    print("test error: %2.2f %%" % test_error)
