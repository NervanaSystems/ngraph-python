#!/usr/bin/env python
# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
from __future__ import division
from __future__ import print_function
from contextlib import closing
import os
import pytest
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Layer, Affine, Preprocess, Sequential
from ngraph.frontends.neon import GaussianInit, Rectlin, Logistic, GradientDescentMomentum
from ngraph.frontends.neon import ax, make_bound_computation
from ngraph.frontends.neon import ArrayIterator

from ngraph.frontends.neon import MNIST
import ngraph.transformers as ngt


def train_mnist_mlp(transformer_name, data_dir=None,
                    rng_seed=12, batch_size=128, train_iter=10, eval_iter=10):
    assert transformer_name in ['cpu', 'hetr']
    assert isinstance(rng_seed, int)

    # Apply this metadata to graph regardless of transformer,
    # but it is ignored for non-HeTr case
    hetr_device_ids = (0, 1)

    # use consistent rng seed between runs
    np.random.seed(rng_seed)

    # Data
    train_data, valid_data = MNIST(path=data_dir).load_data()
    train_set = ArrayIterator(train_data, batch_size, total_iterations=train_iter)
    valid_set = ArrayIterator(valid_data, batch_size)
    inputs = train_set.make_placeholders()
    ax.Y.length = 10

    # Model
    with ng.metadata(device_id=hetr_device_ids, parallel=ax.N):
        seq1 = Sequential([Preprocess(functor=lambda x: x / 255.),
                           Affine(nout=100, weight_init=GaussianInit(), activation=Rectlin()),
                           Affine(axes=ax.Y, weight_init=GaussianInit(), activation=Logistic())])

        train_prob = seq1(inputs['image'])
        train_loss = ng.cross_entropy_binary(train_prob, ng.one_hot(inputs['label'], axis=ax.Y))

        optimizer = GradientDescentMomentum(0.1, 0.9)
        batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
        train_outputs = dict(batch_cost=batch_cost)

        with Layer.inference_mode_on():
            inference_prob = seq1(inputs['image'])
        errors = ng.not_equal(ng.argmax(inference_prob, out_axes=[ax.N]), inputs['label'])
        eval_loss = ng.cross_entropy_binary(inference_prob, ng.one_hot(inputs['label'], axis=ax.Y))
        eval_outputs = dict(cross_ent_loss=eval_loss, misclass_pct=errors)

    # Runtime
    with closing(ngt.make_transformer_factory(transformer_name)()) as transformer:
        train_computation = make_bound_computation(transformer, train_outputs, inputs)
        loss_computation = make_bound_computation(transformer, eval_outputs, inputs)

        train_costs = list()
        for step in range(train_iter):
            out = train_computation(next(train_set))
            train_costs.append(float(out['batch_cost']))

        ce_loss = list()
        for step in range(eval_iter):
            out = loss_computation(next(valid_set))
            ce_loss.append(np.mean(out['cross_ent_loss']))

        return train_costs, ce_loss


@pytest.mark.hetr_only
def test_compare_hetr_cpu():
    """
    Train and eval the same model using CPU transformer and the HeTr
    transformer, comparing the per-batch costs.

    This was added to catch issues with gradient scaling when using
    AllReduce, but may be useful for other reasons and could be extended.
    """
    BASE_DATA_DIR = os.getenv('BASE_DATA_DIR')

    cpu_train_costs, cpu_eval_costs = train_mnist_mlp('cpu', data_dir=BASE_DATA_DIR)
    hetr_train_costs, hetr_eval_costs = train_mnist_mlp('hetr', data_dir=BASE_DATA_DIR)

    ng.testing.assert_allclose(cpu_train_costs, hetr_train_costs)
    ng.testing.assert_allclose(cpu_eval_costs, hetr_eval_costs)
