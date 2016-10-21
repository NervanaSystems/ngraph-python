# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
from __future__ import division

import ngraph as ng
from ngraph.frontends.neon.axis import ax
import numpy as np
from ngraph.transformers import Transformer
import collections


def flatten(item):
    """
    TODO.

    Arguments:
      item: TODO

    Returns:

    """
    if isinstance(item, collections.Iterable):
        for i in iter(item):
            for j in flatten(i):
                yield j
    else:
        yield item


class SimpleModel(object):
    def __init__(self, layers):
        self.layers = layers

    def initialize(self, in_axes):
        for l in self.layers:
            in_axes = l.initialize(in_axes)

    def get_outputs(self, in_obj):
        for l in self.layers:
            in_obj = l.get_outputs(in_obj)
        return in_obj


class ModelTrainer(object):

    def __init__(self, num_iterations, iter_interval):
        self.num_iterations = num_iterations
        self.iter_interval = iter_interval

    def make_train_graph(self, model, inputs, targets, cost_func, optimizer):
        model.initialize(inputs.axes)
        self.train_cost = cost_func(model.get_outputs(inputs), targets)
        self.updates = optimizer(self.train_cost)
        self.mean_cost = ng.mean(self.train_cost, out_axes=())

        self.train_args = ([self.mean_cost, self.updates], inputs, targets)

    def make_inference_graph(self, model, inputs):
        self.inference_args = (model.get_outputs(inputs), inputs)

    def bind_transformer(self, transformer):
        self.train_comp = transformer.computation(*self.train_args)
        self.predictions = transformer.computation(*self.inference_args)
        transformer.initialize()

    def train(self, train_set):
        batch_costs = []
        for mb_idx, dtuple in enumerate(train_set, start=1):
            batch_cost, _ = self.train_comp(dtuple[0]/255., dtuple[1])
            batch_costs.append(float(batch_cost))
            if mb_idx % self.iter_interval == 0:
                print("[Iter %s/%s] Cost = %s" % (mb_idx,
                                                  self.num_iterations, np.mean(batch_costs)))
                batch_costs = []
            if mb_idx >= self.num_iterations:
                return

    def eval(self, eval_set):
        eval_set.reset()
        hyps, refs = [], []
        while len(hyps) < eval_set.ndata:
            dtuple = next(eval_set)
            batch_hyps = np.argmax(self.predictions(dtuple[0]/255.), axis=0)
            bsz = min(eval_set.ndata - len(hyps), len(batch_hyps))
            hyps.extend(list(batch_hyps[:bsz]))
            refs.extend(list(dtuple[1][0][:bsz]))
        return hyps, refs
