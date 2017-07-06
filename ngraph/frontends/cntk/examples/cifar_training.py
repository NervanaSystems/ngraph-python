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
import time
from builtins import range

import cntk as C
import cntk.io.transforms as xforms
import numpy as np
from PIL import Image

import ngraph as ng
from ngraph.frontends.cntk.cntk_importer.importer import (CNTKImporter,
                                                          classification_error,
                                                          cross_entropy_with_softmax)
from ngraph.frontends.common.utils import CommonSGDOptimizer


def create_reader(map_file, mean_file, train):
    """
    Define the reader for both training and evaluation action.
    """

    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("This example require prepared dataset. \
         Please run cifar_prepare.py example.")

    transforms = []
    if train:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.8)
        ]
    transforms += [
        xforms.scale(
            width=image_width,
            height=image_height,
            channels=num_channels,
            interpolations='linear'
        ),
        xforms.mean(mean_file)
    ]

    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        features=C.io.StreamDef(field='image', transforms=transforms),
        labels=C.io.StreamDef(field='label', shape=num_classes)
    )))


def create_basic_model(input, out_dims):
    net = C.layers.Convolution(
        (5, 5), 32, init=C.initializer.glorot_uniform(), activation=C.relu, pad=True
    )(input)
    net = C.layers.MaxPooling((3, 3), strides=(2, 2))(net)

    net = C.layers.Convolution(
        (5, 5), 32, init=C.initializer.glorot_uniform(), activation=C.relu, pad=True
    )(net)
    net = C.layers.MaxPooling((3, 3), strides=(2, 2))(net)

    net = C.layers.Convolution(
        (5, 5), 64, init=C.initializer.glorot_uniform(), activation=C.relu, pad=True
    )(net)
    net = C.layers.MaxPooling((3, 3), strides=(2, 2))(net)

    net = C.layers.Dense(64, init=C.initializer.glorot_uniform())(net)
    net = C.layers.Dense(out_dims, init=C.initializer.glorot_uniform(), activation=None)(net)

    return net


def create_terse_model(input, out_dims):
    with C.layers.default_options(activation=C.relu):
        model = C.layers.Sequential([
            C.layers.For(range(3), lambda i: [
                C.layers.Convolution(
                    (5, 5), [32, 32, 64][i], init=C.initializer.glorot_uniform(), pad=True
                ),
                C.layers.MaxPooling((3, 3), strides=(2, 2))
            ]),
            C.layers.Dense(64, init=C.initializer.glorot_uniform()),
            C.layers.Dense(out_dims, init=C.initializer.glorot_uniform(), activation=None)
        ])

    return model(input)


def create_dropout_model(input, out_dims):
    with C.layers.default_options(activation=C.relu):
        model = C.layers.Sequential([
            C.layers.For(range(3), lambda i: [
                C.layers.Convolution(
                    (5, 5), [32, 32, 64][i], init=C.initializer.glorot_uniform(), pad=True
                ),
                C.layers.MaxPooling((3, 3), strides=(2, 2))
            ]),
            C.layers.Dense(64, init=C.initializer.glorot_uniform()),
            C.layers.Dropout(0.25),
            C.layers.Dense(out_dims, init=C.initializer.glorot_uniform(), activation=None)
        ])

    return model(input)


def create_vgg9_model(input, out_dims):
    with C.layers.default_options(activation=C.relu):
        model = C.layers.Sequential([
            C.layers.For(range(3), lambda i: [
                C.layers.Convolution(
                    (3, 3), [64, 96, 128][i], init=C.initializer.glorot_uniform(), pad=True
                ),
                C.layers.Convolution(
                    (3, 3), [64, 96, 128][i], init=C.initializer.glorot_uniform(), pad=True
                ),
                C.layers.MaxPooling((3, 3), strides=(2, 2))
            ]),
            C.layers.For(range(2), lambda: [
                C.layers.Dense(1024, init=C.initializer.glorot_uniform())
            ]),
            C.layers.Dense(out_dims, init=C.initializer.glorot_uniform(), activation=None)
        ])
    return model(input)


def train_and_evaluate(reader_train, reader_test, max_epochs, model_func):
    """
    Train and evaluate the network.
    """
    input_var = C.input((num_channels, image_height, image_width))
    label_var = C.input((num_classes))

    feature_scale = 1.0 / 256.0
    input_var_norm = C.element_times(feature_scale, input_var)

    model = model_func(input_var_norm, out_dims=10)

    loss = C.cross_entropy_with_softmax(model, label_var)
    error = C.classification_error(model, label_var)

    minibatch_size = 64

    # ngraph model import begin ============================================================
    ng_model, ng_placeholders = CNTKImporter(batch_size=minibatch_size).import_model(model)

    ng_labels = ng.placeholder([ng.make_axis(num_classes), ng.make_axis(minibatch_size, 'N')])
    ng_placeholders.append(ng_labels)

    transformer = ng.transformers.make_transformer()

    ng_loss = cross_entropy_with_softmax(ng_model, ng_labels)
    parallel_update = CommonSGDOptimizer(0.01).minimize(ng_loss, ng_loss.variables())
    training_fun = transformer.computation([ng_loss, parallel_update], *ng_placeholders)

    ng_error = classification_error(ng_model, ng_labels)
    test_fun = transformer.computation(ng_error, *ng_placeholders)
    # ngraph model import end ==============================================================

    # ======================================================================================
    # Training
    # ======================================================================================
    epoch_size = 50000

    # Set training parameters
    lr_per_minibatch = C.learners.learning_rate_schedule(
        [0.01] * 10 + [0.003] * 10 + [0.001],
        C.learners.UnitType.minibatch,
        epoch_size
    )
    momentum_time_constant = C.learners.momentum_as_time_constant_schedule(
        -minibatch_size / np.log(0.9)
    )
    l2_reg_weight = 0.001

    # trainer object
    learner = C.learners.momentum_sgd(
        model.parameters,
        lr=lr_per_minibatch, momentum=momentum_time_constant,
        l2_regularization_weight=l2_reg_weight
    )
    trainer = C.Trainer(model, (loss, error), [learner])

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    # perform model training
    batch_index = 0
    for i in range(max_epochs):
        time_started = time.time()
        sample_count = 0
        while sample_count < epoch_size:
            data = reader_train.next_minibatch(
                min(minibatch_size, epoch_size - sample_count),
                input_map=input_map
            )
            trainer.train_minibatch(data)
            sample_count += data[label_var].num_samples
            batch_index += 1
        print("CNTK epoch {0} - train time: {1} loss: {2:.4f}".format(
            i + 1,
            time.strftime("%H:%M:%S", time.gmtime(time.time() - time_started)),
            trainer.previous_minibatch_loss_average
        ))

    # ngraph model training begin ==========================================================
    features_batch = []
    labels_batch = []
    for i in range(max_epochs):
        time_started = time.time()
        for line in open(os.path.join('/tmp/data/CIFAR-10', 'train_map.txt')):
            if len(features_batch) == minibatch_size:
                ret = training_fun(np.transpose(features_batch), np.transpose(labels_batch))
                features_batch.clear()
                labels_batch.clear()

            image_file, label = line.split()

            image = Image.open(image_file)
            rgb_image = np.asarray(
                image.crop((0, 0, image.size[0] * 0.8, image.size[1] * 0.8)),
                dtype="float32"
            )
            pic = np.ascontiguousarray(rgb_image)
            features_batch.append(pic)

            labels_array = np.zeros(num_classes, dtype="float32")
            labels_array[int(label)] = 1
            labels_batch.append(labels_array)
        print("ngraph epoch {0} - train time: {1} loss: {2:.4f}".format(
            i + 1,
            time.strftime("%H:%M:%S", time.gmtime(time.time() - time_started)),
            float(ret[0])
        ))
    # ngraph model training end ============================================================

    # ======================================================================================
    # Evaluation
    # ======================================================================================
    epoch_size = 10000

    # process minibatches and evaluate the model
    metric_numer = 0
    metric_denom = 0
    sample_count = 0
    while sample_count < epoch_size:
        current_minibatch = min(minibatch_size, epoch_size - sample_count)

        # Fetch next test min batch.
        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)

        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch

        # Keep track of the number of samples processed so far.
        sample_count += data[label_var].num_samples

    print("")
    print("Final CNTK results: {0:.2f}%".format(
        (metric_numer * 100.0) / metric_denom
    ))

    # ngraph model test begin ============================================================
    features_batch.clear()
    labels_batch.clear()
    results = 0.0
    minibatch_index = 0
    for line in open(os.path.join('/tmp/data/CIFAR-10', 'test_map.txt')):
        if len(features_batch) == minibatch_size:
            results += test_fun(np.transpose(features_batch), np.transpose(labels_batch))
            features_batch.clear()
            labels_batch.clear()
            minibatch_index += 1

        image_file, label = line.split()

        rgb_image = np.asarray(Image.open(image_file), dtype="float32")
        pic = np.ascontiguousarray(rgb_image)
        features_batch.append(pic)

        labels_array = np.zeros(num_classes, dtype="float32")
        labels_array[int(label)] = 1
        labels_batch.append(labels_array)

    print("Final ngraph results: {0:.2f}%".format(
        (results * 100.0) / minibatch_index
    ))
    print("")
    # ngraph model test end ==============================================================

    return C.softmax(model)


if __name__ == "__main__":
    np.random.seed(0)
    C.device.try_set_default_device(C.device.cpu())
    factory = ng.transformers.make_transformer_factory('cpu')
    ng.transformers.set_transformer_factory(factory)

    image_height = 32
    image_width = 32
    num_channels = 3
    num_classes = 10

    data_path = os.path.join('/tmp/data', 'CIFAR-10')
    reader_train = create_reader(
        os.path.join(data_path, 'train_map.txt'),
        os.path.join(data_path, 'CIFAR-10_mean.xml'),
        True
    )
    reader_test = create_reader(
        os.path.join(data_path, 'test_map.txt'),
        os.path.join(data_path, 'CIFAR-10_mean.xml'),
        False
    )

    print("Basic model:")
    basic_model = train_and_evaluate(reader_train, reader_test, 5, create_basic_model)
    basic_model.save(data_path + "/basic_model.dnn")

    print("Terse model:")
    terse_model = train_and_evaluate(reader_train, reader_test, 10, create_terse_model)
    terse_model.save(data_path + "/terse_model.dnn")

    print("Dropout model:")
    dropout_model = train_and_evaluate(reader_train, reader_test, 5, create_dropout_model)
    dropout_model.save(data_path + "/dropout_model.dnn")

    print("VGG9 model:")
    vgg9_model = train_and_evaluate(reader_train, reader_test, 5, create_vgg9_model)
    vgg9_model.save(data_path + "/vgg9_model.dnn")
