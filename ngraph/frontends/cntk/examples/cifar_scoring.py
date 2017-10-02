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
Example based on CNTK_201B_CIFAR-10_ImageHandsOn tutorial.
"""
from __future__ import division, print_function

import os

import cntk as C
import numpy as np
from PIL import Image

import ngraph as ng
from ngraph.frontends.cntk.cntk_importer.importer import CNTKImporter


def load_and_score(cifar_dir, model_file):
    map_file = os.path.join(cifar_dir, 'test_map.txt')
    if not os.path.exists(map_file):
        raise RuntimeError("This example require prepared dataset. \
         Please run cifar_prepare.py example.")

    trained_network = os.path.join(cifar_dir, model_file)
    if not os.path.exists(trained_network):
        raise RuntimeError("This example require trained network. \
         Please run cifar_training.py example.")

    cntk_model = C.ops.functions.load_model(trained_network)
    ng_model, ng_placeholders = CNTKImporter().import_model(cntk_model)
    eval_fun = ng.transformers.make_transformer().computation(ng_model, *ng_placeholders)

    cntk_results = []
    ng_results = []
    same_predictions = []
    for line in open(map_file):
        image_file, label = line.split()

        rgb_image = np.asarray(Image.open(image_file), dtype="float32")
        pic = np.ascontiguousarray(np.moveaxis(rgb_image, 2, 0))

        cntk_predictions = np.squeeze(
            cntk_model.eval({cntk_model.arguments[0]: [pic]})
        )
        cntk_results.append(int(np.argmax(cntk_predictions)) == int(label))

        ng_predictions = eval_fun(pic)
        ng_results.append(int(np.argmax(ng_predictions)) == int(label))

        same_predictions.append(np.allclose(cntk_predictions, ng_predictions))

    test_size = len(ng_results)
    if np.all(same_predictions):
        print('CNTK and ngraph predictions identical. Prediction correctness - {0:.2f}%.'.format(
            np.count_nonzero(ng_results) * 100 / test_size
        ))
    else:
        print('CNTK prediction correctness - {0:.2f}%'.format(
            np.count_nonzero(cntk_results) * 100 / test_size
        ))
        print('ngraph prediction correctness - {0:.2f}%'.format(
            np.count_nonzero(cntk_results) * 100 / test_size
        ))
    print("")


if __name__ == "__main__":
    np.random.seed(0)
    data_dir = os.path.join('/tmp/data/', 'CIFAR-10')

    print("Basic model:")
    load_and_score(data_dir, "basic_model.dnn")

    print("Terse model:")
    load_and_score(data_dir, "terse_model.dnn")

    print("Dropout model:")
    load_and_score(data_dir, "dropout_model.dnn")

    print("VGG9 model:")
    load_and_score(data_dir, "vgg9_model.dnn")

    print("ResNet model:")
    load_and_score(data_dir, "resnet_model.dnn")
