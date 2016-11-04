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

import numpy as np
from itertools import takewhile
import functools

class Model(object):
    def __init__(self, layers):
        self.layers = layers

    def get_outputs(self, in_obj):
        for l in self.layers:
            in_obj = l.train_outputs(in_obj)
        return in_obj

    def bind_transformer(self, transformer, train_args, inference_args):
        self.train_comp = transformer.computation(*train_args)
        self.predictions = transformer.computation(*inference_args)
        transformer.initialize()

    def train(self, train_set, num_iterations, callbacks):
        callbacks.on_train_begin(num_iterations)

        for mb_idx, dtuple in takewhile(lambda x: x[0] < num_iterations, enumerate(train_set)):
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


class Container(object):
    """
    Two string->`Op` dictionaries representing a container of op_graphs
    """
    def __init__(self, inputs=dict(), outputs=dict()):
        self.inputs = inputs
        self.outputs = outputs

    def add(self, rhs):
        new_inputs = self.inputs.copy()
        new_outputs = self.outputs.copy()
        # these label -> Op mappings are
        # still pointing to the same ops
        new_inputs.update(rhs.inputs)
        new_outputs.update(rhs.outputs)
        return Container(new_inputs, new_outputs)

    def subset(self, inputs=None, outputs=None):
        """
        Eventually, a user should be able to subset using op_graph `Selectors`
        similar to XPath or Jquery selectors. Boolean combinations of op type,
        op name regex, and upstream, downstream from given ops.
        Here we just have selection by exact name.
        """
        return Container({k: v for k,v in self.inputs.items() if k in inputs},
                         {k: v for k,v in self.outputs.items() if k in outputs})


def ne_compose(layer_list, function_name):

    inputs, outputs = dict(), dict()

    out_func = functools.reduce(lambda f, g:
                         lambda x: f(getattr(g, function_name)(x)),
                     layer_list,
                     lambda x: x)
    outputs[getattr(layer_list[-1], 'outputs')] = out_func

    return Container(inputs=inputs, outputs=outputs)


