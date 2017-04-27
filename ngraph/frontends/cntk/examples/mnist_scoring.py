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
from __future__ import print_function

import glob
import os

import cntk as C
import numpy as np
from PIL import Image

import ngraph as ng
from ngraph.frontends.cntk.cntk_importer.importer import CNTKImporter


def load_and_score(mnist_jpg_dir):
    cntk_model = C.ops.functions.load_model("/tmp/data/MNIST/MNIST.dnn")

    ng_model, ng_placeholders = CNTKImporter().import_model(cntk_model)
    eval_fun = ng.transformers.make_transformer().computation(ng_model, *ng_placeholders)

    for filename in glob.glob(os.path.join(mnist_jpg_dir, '*.jpg')):
        rgb_image = np.asarray(Image.open(filename), dtype="float32")
        pic = np.ascontiguousarray(rgb_image).flatten()

        cntk_predictions = np.squeeze(
            cntk_model.eval({cntk_model.arguments[0]: [pic]}))
        cntk_top_class = np.argmax(cntk_predictions)

        ng_predictions = eval_fun(pic)
        ng_top_class = np.argmax(ng_predictions)

        actual_number = os.path.basename(filename).split("_")[1]
        print("Digit in jpg file: " + actual_number)
        print("\tCNTK prediction:   " + str(cntk_top_class))
        print("\tngraph prediction: " + str(ng_top_class))
        print("")


if __name__ == "__main__":
    load_and_score("/tmp/data/MNIST/jpg/")
