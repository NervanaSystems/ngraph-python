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
from builtins import range

import cntk as C
import cntk.io.transforms as xforms
import numpy as np

import ngraph as ng
from ngraph.frontends.cntk.cntk_importer.importer import (CNTKImporter,
                                                          classification_error,
                                                          create_loss_and_learner)


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


def convolution_bn(input, filter_size, num_filters, strides=(1, 1),
                   init=C.he_normal(), activation=C.relu):
    if activation is None:
        activation = lambda x: x

    r = C.layers.Convolution(
        filter_size, num_filters,
        strides=strides, init=init,
        activation=None, pad=True, bias=False
    )(input)
    # r = C.layers.BatchNormalization(map_rank=1)(r)
    return activation(r)


def resnet_basic(input, num_filters):
    c1 = convolution_bn(input, (3, 3), num_filters)
    c2 = convolution_bn(c1, (3, 3), num_filters, activation=None)
    return C.relu(c2 + input)


def resnet_basic_inc(input, num_filters):
    c1 = convolution_bn(input, (3, 3), num_filters, strides=(2, 2))
    c2 = convolution_bn(c1, (3, 3), num_filters, activation=None)
    s = convolution_bn(input, (1, 1), num_filters, strides=(2, 2), activation=None)
    return C.relu(c2 + s)


def resnet_basic_stack(input, num_filters, num_stack):
    assert num_stack > 0
    r = input
    for _ in range(num_stack):
        r = resnet_basic(r, num_filters)
    return r


def create_resnet_model(input, out_dims):
    conv = convolution_bn(input, (3, 3), 16)
    r1_1 = resnet_basic_stack(conv, 16, 3)

    r2_1 = resnet_basic_inc(r1_1, 32)
    r2_2 = resnet_basic_stack(r2_1, 32, 2)

    r3_1 = resnet_basic_inc(r2_2, 64)
    r3_2 = resnet_basic_stack(r3_1, 64, 2)

    pool = C.layers.AveragePooling(filter_shape=(8, 8), strides=(1, 1))(r3_2)
    net = C.layers.Dense(out_dims, init=C.he_normal(), activation=None)(pool)
    return net


def train_and_evaluate(reader_train, reader_test, max_epochs, model_func):
    # ======================================================================================
    # Creating
    # ======================================================================================
    input_var = C.input((num_channels, image_height, image_width))
    feature_scale = 1.0 / 256.0
    input_var_norm = C.element_times(feature_scale, input_var)

    cntk_model = model_func(input_var_norm, num_classes)

    label_var = C.input((num_classes))
    loss = C.cross_entropy_with_softmax(cntk_model, label_var)
    error = C.classification_error(cntk_model, label_var)

    minibatch_size = 64
    learning_rate = 0.01
    momentum = 0.9

    lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
    learner = C.momentum_sgd(cntk_model.parameters, lr_schedule, C.momentum_schedule(momentum))
    trainer = C.Trainer(cntk_model, (loss, error), [learner])

    ng_model, ng_placeholders = CNTKImporter(batch_size=minibatch_size).import_model(cntk_model)
    ng_labels = ng.placeholder([ng.make_axis(num_classes), ng.make_axis(minibatch_size, 'N')])
    ng_placeholders.append(ng_labels)

    transformer = ng.transformers.make_transformer()

    ng_loss = create_loss_and_learner(ng_model, ng_labels, learning_rate, momentum)
    training_fun = transformer.computation(ng_loss, *ng_placeholders)

    ng_error = classification_error(ng_model, ng_labels)
    test_fun = transformer.computation(ng_error, *ng_placeholders)

    # ======================================================================================
    # Training
    # ======================================================================================
    epoch_size = 50000
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    num_minibatches_to_train = (epoch_size * max_epochs) / minibatch_size
    for _ in range(0, int(num_minibatches_to_train)):
        data = reader_train.next_minibatch(minibatch_size, input_map=input_map)
        trainer.train_minibatch(data)

        features_batch = np.moveaxis(np.squeeze(data[input_var].asarray()), 0, -1)
        labels_batch = np.moveaxis(
            data[label_var].data.data.slice_view(
                [0, 0, 0], [minibatch_size, num_classes]
            ).asarray().todense(),
            0, -1
        )
        training_fun(features_batch, labels_batch)

    # ======================================================================================
    # Evaluation
    # ======================================================================================
    cntk_results = 0.0
    ng_results = 0.0
    epoch_size = 10000
    input_map = {
        input_var: reader_test.streams.features,
        label_var: reader_test.streams.labels
    }

    num_minibatches_to_test = epoch_size // minibatch_size
    for _ in range(num_minibatches_to_test):
        data = reader_test.next_minibatch(minibatch_size, input_map=input_map)
        cntk_results += trainer.test_minibatch(data)

        features_batch = np.moveaxis(np.squeeze(data[input_var].asarray()), 0, -1)
        labels_batch = np.moveaxis(
            data[label_var].data.data.slice_view([0, 0, 0], [64, 10]).asarray().todense(),
            0, -1
        )
        ng_results += test_fun(features_batch, labels_batch)

    print("CNTK results: {0:.2f}%".format((cntk_results * 100.0) / num_minibatches_to_test))
    print("ngraph results: {0:.2f}%".format((ng_results * 100.0) / num_minibatches_to_test))
    print("")

    return C.softmax(cntk_model)


if __name__ == "__main__":
    np.random.seed(0)

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
    terse_model = train_and_evaluate(reader_train, reader_test, 5, create_terse_model)
    terse_model.save(data_path + "/terse_model.dnn")

    print("Dropout model:")
    dropout_model = train_and_evaluate(reader_train, reader_test, 5, create_dropout_model)
    dropout_model.save(data_path + "/dropout_model.dnn")

    print("VGG9 model:")
    vgg9_model = train_and_evaluate(reader_train, reader_test, 5, create_vgg9_model)
    vgg9_model.save(data_path + "/vgg9_model.dnn")

    print("ResNet model:")
    resnet_model = train_and_evaluate(reader_train, reader_test, 5, create_resnet_model)
    resnet_model.save(data_path + "/resnet_model.dnn")
