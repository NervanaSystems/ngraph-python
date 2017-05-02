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
from __future__ import print_function
from __future__ import division

import os
import gzip
import struct
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

import cntk as C
import numpy as np

import ngraph as ng
from ngraph.frontends.cntk.cntk_importer.importer import CNTKImporter, \
    cross_entropy_with_softmax, classification_error
from ngraph.frontends.common.utils import CommonSGDOptimizer


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


def downloadMNIST(data_dir):
    url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    num_train_samples = 60000
    train = try_download(url_train_image, url_train_labels, num_train_samples)
    url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    num_test_samples = 10000
    test = try_download(url_test_image, url_test_labels, num_test_samples)

    savetxt(os.path.join(data_dir, "Train-28x28_cntk_text.txt"), train)
    savetxt(os.path.join(data_dir, "Test-28x28_cntk_text.txt"), test)


def create_reader(path, is_training, input_dim, num_label_classes):
    return C.io.MinibatchSource(
        C.io.CTFDeserializer(
            path,
            C.io.StreamDefs(
                labels=C.io.StreamDef(
                    field='labels',
                    shape=num_label_classes,
                    is_sparse=False
                ),
                features=C.io.StreamDef(
                    field='features',
                    shape=input_dim,
                    is_sparse=False
                )
            )
        ),
        randomize=is_training,
        epoch_size=C.io.INFINITELY_REPEAT if is_training else C.io.FULL_DATA_SWEEP
    )


def create_model(features, num_hidden_layers, hidden_layers_dim, num_output_classes):
    with C.layers.default_options(init=C.initializer.glorot_uniform(), activation=C.ops.relu):
        h = features
        for _ in range(num_hidden_layers):
            h = C.layers.Dense(hidden_layers_dim)(h)
        r = C.layers.Dense(num_output_classes, activation=None)(h)
        return r


def train_and_test(data_dir):
    input_dim = 784
    num_output_classes = 10

    train_file = os.path.join(data_dir, "Train-28x28_cntk_text.txt")
    test_file = os.path.join(data_dir, "Test-28x28_cntk_text.txt")

    num_hidden_layers = 2
    hidden_layers_dim = 400

    input = C.input(input_dim)
    label = C.input(num_output_classes)

    z = create_model(input / 256.0, num_hidden_layers, hidden_layers_dim, num_output_classes)

    loss = C.cross_entropy_with_softmax(z, label)
    label_error = C.classification_error(z, label)

    learning_rate = 0.2
    lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
    learner = C.sgd(z.parameters, lr_schedule)
    trainer = C.Trainer(z, (loss, label_error), [learner])

    batch_size = 64
    num_samples_per_sweep = 60000
    num_sweeps_to_train_with = 10
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / batch_size

    # ngraph import begin ==================================================================
    ng_model, ng_placeholders = CNTKImporter(batch_size=batch_size).import_model(z)

    ng_labels = ng.placeholder([ng.make_axis(num_output_classes), ng.make_axis(batch_size, 'N')])
    ng_placeholders.append(ng_labels)

    ng_loss = cross_entropy_with_softmax(ng_model, ng_labels)
    parallel_update = CommonSGDOptimizer(learning_rate).minimize(ng_loss, ng_loss.variables())

    transformer = ng.transformers.make_transformer()
    training_fun = transformer.computation([ng_loss, parallel_update], *ng_placeholders)

    ng_error = classification_error(ng_model, ng_labels)
    test_fun = transformer.computation(ng_error, *ng_placeholders)
    # ngraph import end ====================================================================

    reader_train = create_reader(train_file, True, input_dim, num_output_classes)

    input_map = {
        label: reader_train.streams.labels,
        input: reader_train.streams.features
    }

    for _ in range(0, int(num_minibatches_to_train)):
        data = reader_train.next_minibatch(batch_size, input_map=input_map)
        trainer.train_minibatch(data)

    reader_test = create_reader(test_file, False, input_dim, num_output_classes)

    test_input_map = {
        label: reader_test.streams.labels,
        input: reader_test.streams.features,
    }

    num_samples = 10000
    num_minibatches_to_test = num_samples // batch_size
    test_result = 0.0

    for _ in range(num_minibatches_to_test):
        data = reader_test.next_minibatch(batch_size, input_map=test_input_map)
        test_result += trainer.test_minibatch(data)

    print("Average CNTK test error: {0:.2f}%".format(test_result * 100 / num_minibatches_to_test))
    z.save_model(data_dir + "MNIST.dnn")

    # ngraph batch training begin ============================================================
    features_batch = []
    labels_batch = []
    for _ in range(0, num_sweeps_to_train_with):
        for line in open(train_file):
            if len(features_batch) == batch_size:
                training_fun(np.transpose(features_batch), np.transpose(labels_batch))
                features_batch.clear()
                labels_batch.clear()

            minibatch_data = line.split()

            labels = minibatch_data[
                minibatch_data.index("|labels") + 1:minibatch_data.index("|features")
            ]
            labels = [float(i) for i in labels]
            labels_batch.append(labels)

            features = minibatch_data[
                minibatch_data.index("|features") + 1:
            ]
            features = [float(i) for i in features]
            features_batch.append(features)
    # ngraph batch training end ==============================================================

    # ngraph batch testing begin =============================================================
    features_batch.clear()
    labels_batch.clear()
    ng_error = 0.0
    for line in open(test_file):
        if len(features_batch) == batch_size:
            ng_error += test_fun(np.transpose(features_batch), np.transpose(labels_batch))
            features_batch.clear()
            labels_batch.clear()

        minibatch_data = line.split()

        labels = minibatch_data[
            minibatch_data.index("|labels") + 1:minibatch_data.index("|features")
        ]
        labels = [float(i) for i in labels]
        labels_batch.append(labels)

        features = minibatch_data[
            minibatch_data.index("|features") + 1:
        ]
        features = [float(i) for i in features]
        features_batch.append(features)
    print("Average ngraph test error: {0:.2f}%".format(ng_error * 100 / num_minibatches_to_test))
    # ngraph batch testing end ===============================================================


if __name__ == "__main__":
    data_dir = os.path.join("/tmp/data", "MNIST")
    downloadMNIST(data_dir)

    np.random.seed(0)
    train_and_test(data_dir)
