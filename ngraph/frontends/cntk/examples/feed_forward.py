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
Example based on CNTK_102_FeedForward tutorial.
"""
from __future__ import print_function

import cntk as C
import numpy as np

import ngraph as ng
from ngraph.frontends.cntk.cntk_importer.importer import CNTKImporter
from ngraph.frontends.common.utils import CommonSGDOptimizer


def generate_random_sample(sample_size, feature_dim, num_classes):
    """
    Helper function to generate a random data sample.
    """
    X = np.random.normal(loc=0.0, scale=0.5, size=(sample_size, feature_dim)).astype(np.float32)

    C = np.random.randint(num_classes, size=(sample_size))
    Y = np.zeros((sample_size, num_classes))
    Y[np.arange(sample_size), C] = 1

    return X, Y.astype(np.float32)


def linear_layer(input_var, output_dim):
    input_dim = input_var.shape[0]

    weight = C.parameter(shape=(input_dim, output_dim))
    bias = C.parameter(shape=(output_dim))

    return bias + C.times(input_var, weight)


def dense_layer(input_var, output_dim, nonlinearity):
    l = linear_layer(input_var, output_dim)
    return nonlinearity(l)


def fully_connected_classifier_net(input_var, num_output_classes,
                                   hidden_layer_dim, num_hidden_layers,
                                   nonlinearity):

    h = dense_layer(input_var, hidden_layer_dim, nonlinearity)
    for i in range(1, num_hidden_layers):
        h = dense_layer(h, hidden_layer_dim, nonlinearity)

    return linear_layer(h, num_output_classes)


if __name__ == "__main__":
    np.random.seed(0)

    input_dim = 2
    num_output_classes = 2
    sample_size = 32
    number_of_iterations = 10

    features, labels = generate_random_sample(sample_size, input_dim, num_output_classes)

    # CNTK
    num_hidden_layers = 2
    hidden_layers_dim = 50

    input = C.input(input_dim)
    label = C.input(num_output_classes)

    z = fully_connected_classifier_net(input, num_output_classes,
                                       hidden_layers_dim, num_hidden_layers,
                                       C.sigmoid)
    loss = C.cross_entropy_with_softmax(z, label)
    eval_error = C.classification_error(z, label)

    learning_rate = 0.5
    lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.sample)

    learner = C.sgd(z.parameters, lr_schedule)
    trainer = C.Trainer(z, (loss, eval_error), [learner])

    # ngraph
    L, placeholders = CNTKImporter().import_model(loss)
    parallel_update = CommonSGDOptimizer(learning_rate).minimize(L, L.variables())

    transformer = ng.transformers.make_transformer()
    update_fun = transformer.computation([L, parallel_update], *placeholders)

    # CNTK training
    for i in range(0, number_of_iterations):
        for xs, ys in zip(features, labels):
            trainer.train_minibatch({input: [xs], label: [ys]})
        training_loss = trainer.previous_minibatch_loss_average
        print("cntk iteration {0} -> loss: {1:.5f}".format(i, training_loss))

    # ngraph training
    for i in range(0, number_of_iterations):
        for xs, ys in zip(features, labels):
            ret = update_fun(xs, ys)
        print("ngraph iteration {0} -> loss: {1:.5f}".format(i, float(ret[0])))
