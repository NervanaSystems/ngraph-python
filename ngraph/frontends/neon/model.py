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

from operator import itemgetter


class Sequential(object):
    def __init__(self, layers):
        self.layers = layers

    def train_outputs(self, in_obj):
        for l in self.layers:
            in_obj = l.train_outputs(in_obj)
        return in_obj

    def inference_outputs(self, in_obj):
        for l in self.layers:
            in_obj = l.inference_outputs(in_obj)
        return in_obj


class Container(object):
    """
    POC code only

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
        return Container({k: v for k, v in self.inputs.items() if k in inputs},
                         {k: v for k, v in self.outputs.items() if k in outputs})


class BoundComputation(object):
    """
    Callable object that has inputs and outputs of a computation bound via names
    """
    def __init__(self, transformer, named_outputs, named_inputs):
        self.input_keys = tuple(named_inputs.keys())
        self.output_keys = tuple(named_outputs.keys())

        outputs = itemgetter(*self.output_keys)(named_outputs)
        outputs = [outputs] if len(self.output_keys) == 1 else list(outputs)

        inputs = itemgetter(*self.input_keys)(named_inputs)
        self.num_outputs = len(outputs)
        self.comp_func = transformer.computation(outputs, *inputs)

    def __call__(self, named_buffers):
        result_tuple = self.comp_func(*itemgetter(*self.input_keys)(named_buffers))
        result_dict = {k: v for k, v in zip(self.output_keys, result_tuple)}
        return result_dict


def make_bound_computation(transformer, named_outputs, named_inputs):
    return BoundComputation(transformer, named_outputs, named_inputs)
