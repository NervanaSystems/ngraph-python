# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
Example based on network trained by mnist_training.py example. It will create
"MNIST.dnn" file in "/tmp/data/MNIST" folder (where MNIST data is stored).
To be able to run this example you need to download the MNST database as jpg
images and put it in /tmp/data/MNIST/jpg/ folder.
"""
from __future__ import division, print_function

import glob
import os

import cntk as C
import numpy as np
from PIL import Image

import ngraph as ng
from ngraph.frontends.cntk.cntk_importer.importer import CNTKImporter


def load_and_score(mnist_dir):
    jpg_files = glob.glob(os.path.join(mnist_dir, 'jpg/*.jpg'))
    if not jpg_files:
        raise RuntimeError(
            "This example require a dataset. Please download and extract the MNIST jpg files."
        )

    trained_network = os.path.join(mnist_dir, "MNIST.dnn")
    if not os.path.exists(trained_network):
        raise RuntimeError(
            "This example require trained network. Please run mnist_training.py example."
        )

    cntk_model = C.ops.functions.load_model(trained_network)

    ng_model, ng_placeholders = CNTKImporter().import_model(cntk_model)
    eval_fun = ng.transformers.make_transformer().computation(ng_model, *ng_placeholders)

    cntk_results = []
    ng_results = []
    for filename in jpg_files:
        label = int(os.path.basename(filename).split("_")[1])

        rgb_image = np.asarray(Image.open(filename), dtype="float32")
        pic = np.ascontiguousarray(rgb_image).flatten()

        cntk_predictions = np.squeeze(
            cntk_model.eval({cntk_model.arguments[0]: [pic]})
        )
        cntk_results.append(int(np.argmax(cntk_predictions) == label))

        ng_predictions = eval_fun(pic)
        ng_results.append(int(np.argmax(ng_predictions) == label))

    test_size = len(jpg_files)
    print('CNTK prediction correctness - {0:.2f}%'.format(
        np.count_nonzero(cntk_results) * 100 / test_size
    ))
    print('ngraph prediction correctness - {0:.2f}%'.format(
        np.count_nonzero(ng_results) * 100 / test_size
    ))


if __name__ == "__main__":
    load_and_score("/tmp/data/MNIST/")
