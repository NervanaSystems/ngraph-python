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
import numpy as np
from itertools import takewhile

class Model(object):
    def __init__(self, layers):
        self.layers = layers
        self.initialized = False

    def initialize(self, in_axes):
        if not self.initialized:
            for l in self.layers:
                in_axes = l.initialize(in_axes)
        self.initialized = True

    def get_outputs(self, in_obj):
        self.initialize(in_obj.axes)
        for l in self.layers:
            in_obj = l.get_outputs(in_obj)
        return in_obj

    def bind_transformer(self, transformer, train_args, inference_args):
        self.train_comp = transformer.computation(*train_args)
        self.predictions = transformer.computation(*inference_args)
        transformer.initialize()

    def train(self, train_set, num_iterations, callbacks):
        callbacks.on_train_begin(num_iterations)

        for mb_idx, dtuple in takewhile(lambda x: x[0]<num_iterations, enumerate(train_set)):
            callbacks.on_minibatch_begin(mb_idx)

            batch_cost, _ = self.train_comp(dtuple[0], dtuple[1], mb_idx)
            self.current_batch_cost = float(batch_cost)

            callbacks.on_minibatch_end(mb_idx)

        callbacks.on_train_end()

    def eval(self, eval_set):
        eval_set.reset()
        hyps, refs = [], []
        while len(hyps) < eval_set.ndata:
            dtuple = next(eval_set)
            batch_hyps = np.argmax(self.predictions(dtuple[0]), axis=1)
            bsz = min(eval_set.ndata - len(hyps), len(batch_hyps))
            hyps.extend(list(batch_hyps[:bsz]))
            refs.extend(list(dtuple[1][0][:bsz]))
        return hyps, refs

