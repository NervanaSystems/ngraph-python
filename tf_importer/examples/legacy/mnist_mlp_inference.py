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
TODO: load meta info from TF's MetaGraph including details about dataset,
      training epochs and etc
"""

from __future__ import print_function, division

import ngraph.transformers as ngt
import ngraph.analysis as an
import numpy as np
from neon.data import MNIST
from neon.util.argparser import NeonArgparser
from tf_importer.tf_importer.importer import TFImporter

# args and environment
parser = NeonArgparser(__doc__)
parser.add_argument('--pb_file', type=str, default="graph_froze.pb",
                    help='GraphDef protobuf')
args = parser.parse_args()


def inference_mnist_mlp():
    """
    Runs mnist mlp inference example

    Returns:
        None
    """

    # data loader
    mnist_data = MNIST(path=args.data_dir).gen_iterators()
    test_data = mnist_data['valid']

    # build graph
    tf_importer = TFImporter(args.pb_file)

    # init transformer
    transformer = ngt.make_transformer()
    infer_computation = transformer.computation(tf_importer.last_op,
                                                tf_importer.x)

    # set ups
    total_error = 0
    total_num_samples = 0

    # visualize
    dataflow = an.DataFlowGraph(transformer, [tf_importer.last_op])
    # dataflow.render('dataflow.gv')
    fused = an.KernelFlowGraph(dataflow, fusible=an.gpu_fusible)
    # fused.render('fused-dataflow.gv')
    interference = an.InterferenceGraph(fused.liveness())
    interference.color()
    # interference.render('interference.gv')

    for batch_idx, (x_raw, y_raw) in enumerate(test_data):
        # increment total number of samples
        num_samples = x_raw.shape[1]
        total_num_samples += num_samples

        # get inference result
        result = infer_computation(x_raw)

        # get errors
        pred = np.argmax(result, axis=1)
        gt = np.argmax(y_raw, axis=0)
        batch_error = float(np.sum(np.not_equal(pred, gt)))
        batch_error_rate = batch_error / float(num_samples) * 100.
        total_error += batch_error

        print("batch index: %d" % batch_idx)
        print("prediction result: %s" % result)
        print("shape of the prediction: %s" % (result.shape,))
        print("batch error: %2.2f" % batch_error_rate)
        print("-------------------------------")

    total_error_rate = total_error / float(total_num_samples) * 100
    print("total error: %2.2f %%" % total_error_rate)


if __name__ == '__main__':
    inference_mnist_mlp()
