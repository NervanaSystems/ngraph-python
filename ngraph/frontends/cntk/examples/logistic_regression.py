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
Example based on CNTK_101_LogisticRegression tutorial.
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

    weight_param = C.parameter(shape=(input_dim, output_dim))
    bias_param = C.parameter(shape=(output_dim))

    return C.times(input_var, weight_param) + bias_param


if __name__ == "__main__":
    np.random.seed(0)

    input_dim = 2
    num_output_classes = 2
    sample_size = 32
    number_of_iterations = 10

    features, labels = generate_random_sample(sample_size, input_dim, num_output_classes)

    # CNTK
    input = C.input_variable(input_dim, np.float32)

    output_dim = num_output_classes
    z = linear_layer(input, output_dim)

    label = C.input_variable((num_output_classes), np.float32)
    loss = C.ops.cross_entropy_with_softmax(z, label)
    eval_error = C.ops.classification_error(z, label)

    learning_rate = 0.05
    lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.sample)

    learner = C.learner.sgd(z.parameters, lr_schedule)
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
        training_loss = C.utils.get_train_loss(trainer)
        print("cntk iteration {0} -> loss: {1}".format(i, training_loss))

    # ngraph training
    for i in range(0, number_of_iterations):
        for xs, ys in zip(features, labels):
            ret = update_fun(xs, ys)
        print("ngraph iteration {0} -> loss: {1}".format(i, ret[0]))
