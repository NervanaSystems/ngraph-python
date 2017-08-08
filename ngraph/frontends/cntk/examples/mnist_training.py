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
Example based on CNTK_103A_MNIST_DataLoader and CNTK_103B_MNIST_FeedForwardNetwork tutorials.
"""
from __future__ import division, print_function

import gzip
import os
import struct

import cntk as C
import numpy as np

import ngraph as ng
from ngraph.frontends.cntk.cntk_importer.importer import (CNTKImporter,
                                                          classification_error,
                                                          cross_entropy_with_softmax)
from ngraph.frontends.common.utils import CommonSGDOptimizer

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


def loadData(src, cimg):
    gzfname, h = urlretrieve(src, './delete.me')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            if n[0] != 0x3080000:
                raise Exception('Invalid file: unexpected magic number.')
            n = struct.unpack('>I', gz.read(4))[0]
            if n != cimg:
                raise Exception('Invalid file: expected {0} entries.'.format(cimg))
            crow = struct.unpack('>I', gz.read(4))[0]
            ccol = struct.unpack('>I', gz.read(4))[0]
            if crow != 28 or ccol != 28:
                raise Exception('Invalid file: expected 28 rows/cols per image.')
            res = np.fromstring(gz.read(cimg * crow * ccol), dtype=np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, crow * ccol))


def loadLabels(src, cimg):
    gzfname, h = urlretrieve(src, './delete.me')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            if n[0] != 0x1080000:
                raise Exception('Invalid file: unexpected magic number.')
            n = struct.unpack('>I', gz.read(4))
            if n[0] != cimg:
                raise Exception('Invalid file: expected {0} rows.'.format(cimg))
            res = np.fromstring(gz.read(cimg), dtype=np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, 1))


def try_download(dataSrc, labelsSrc, cimg):
    data = loadData(dataSrc, cimg)
    labels = loadLabels(labelsSrc, cimg)
    return np.hstack((data, labels))


def savetxt(filename, ndarray):
    dir = os.path.dirname(filename)

    if not os.path.exists(dir):
        os.makedirs(dir)

    if not os.path.isfile(filename):
        with open(filename, 'w') as f:
            labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
            for row in ndarray:
                row_str = row.astype(str)
                label_str = labels[row[-1]]
                feature_str = ' '.join(row_str[:-1])
                f.write('|labels {} |features {}\n'.format(label_str, feature_str))
    else:
        print("File already exists", filename)


def download_and_save(data_dir):
    url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    num_train_samples = 60000
    train = try_download(url_train_image, url_train_labels, num_train_samples)
    savetxt(os.path.join(data_dir, "Train-28x28_cntk_text.txt"), train)

    url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    num_test_samples = 10000
    test = try_download(url_test_image, url_test_labels, num_test_samples)
    savetxt(os.path.join(data_dir, "Test-28x28_cntk_text.txt"), test)


def create_reader(path, is_training, input_dim, output_dim):
    featureStream = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)
    labelStream = C.io.StreamDef(field='labels', shape=output_dim, is_sparse=False)

    return C.io.MinibatchSource(
        C.io.CTFDeserializer(
            path,
            C.io.StreamDefs(labels=labelStream, features=featureStream)
        ),
        randomize=is_training,
        max_sweeps=C.io.INFINITELY_REPEAT if is_training else 1
    )


def create_model(features, num_hidden_layers, hidden_layers_dim, output_dim):
    with C.layers.default_options(init=C.layers.glorot_uniform(), activation=C.ops.relu):
        h = features
        for _ in range(num_hidden_layers):
            h = C.layers.Dense(hidden_layers_dim)(h)
        return C.layers.Dense(output_dim, activation=None)(h)


def train_and_test(data_dir):
    train_file = os.path.join(data_dir, "Train-28x28_cntk_text.txt")
    test_file = os.path.join(data_dir, "Test-28x28_cntk_text.txt")

    input_dim = 784
    output_dim = 10

    input_var = C.input(input_dim)
    label_var = C.input(output_dim)

    cntk_model = create_model(input_var / 256.0, 2, 400, output_dim)

    cntk_loss = C.cross_entropy_with_softmax(cntk_model, label_var)
    cntk_error = C.classification_error(cntk_model, label_var)

    learning_rate = 0.2
    lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
    learner = C.sgd(cntk_model.parameters, lr_schedule)
    trainer = C.Trainer(cntk_model, (cntk_loss, cntk_error), [learner])

    batch_size = 64

    # ngraph import begin ==================================================================
    ng_model, ng_placeholders = CNTKImporter(batch_size=batch_size).import_model(cntk_model)

    ng_labels = ng.placeholder([ng.make_axis(output_dim), ng.make_axis(batch_size, 'N')])
    ng_placeholders.append(ng_labels)

    transformer = ng.transformers.make_transformer()

    ng_loss = cross_entropy_with_softmax(ng_model, ng_labels)
    parallel_update = CommonSGDOptimizer(learning_rate).minimize(ng_loss, ng_loss.variables())
    training_fun = transformer.computation([ng_loss, parallel_update], *ng_placeholders)

    ng_error = classification_error(ng_model, ng_labels)
    test_fun = transformer.computation(ng_error, *ng_placeholders)
    # ngraph import end ====================================================================

    reader_train = create_reader(train_file, True, input_dim, output_dim)
    train_input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    num_samples = 60000
    num_epochs = 10
    num_minibatches_to_train = (num_samples * num_epochs) / batch_size
    for _ in range(0, int(num_minibatches_to_train)):
        data = reader_train.next_minibatch(batch_size, input_map=train_input_map)
        trainer.train_minibatch(data)

        # ngraph train
        features_batch = np.moveaxis(np.squeeze(data[input_var].asarray()), 0, -1)
        labels_batch = np.moveaxis(np.squeeze(data[label_var].asarray()), 0, -1)
        training_fun(features_batch, labels_batch)

    reader_test = create_reader(test_file, False, input_dim, output_dim)
    test_input_map = {
        input_var: reader_test.streams.features,
        label_var: reader_test.streams.labels
    }

    cntk_result = 0.0
    ng_error = 0.0
    num_samples = 10000
    num_minibatches_to_test = num_samples // batch_size
    for _ in range(num_minibatches_to_test):
        data = reader_test.next_minibatch(batch_size, input_map=test_input_map)
        cntk_result += trainer.test_minibatch(data)

        # ngraph test
        features_batch = np.moveaxis(np.squeeze(data[input_var].asarray()), 0, -1)
        labels_batch = np.moveaxis(np.squeeze(data[label_var].asarray()), 0, -1)
        ng_error += test_fun(features_batch, labels_batch)

    print("Average CNTK test error: {0:.2f}%".format(cntk_result * 100 / num_minibatches_to_test))
    print("Average ngraph test error: {0:.2f}%".format(ng_error * 100 / num_minibatches_to_test))

    C.softmax(cntk_model).save(os.path.join(MNIST, "MNIST.dnn"))


if __name__ == "__main__":
    np.random.seed(0)
    MNIST = os.path.join("/tmp/data", "MNIST")
    download_and_save(MNIST)
    train_and_test(MNIST)
