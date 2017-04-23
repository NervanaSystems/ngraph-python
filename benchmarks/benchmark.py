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

from collections import OrderedDict
import ngraph.transformers as ngt
import ngraph.transformers.passes.nviz
import numpy as np
import time
from contextlib import closing


class DefaultOrderedDict(OrderedDict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


class Benchmark(object):

    def __init__(self, computation, train_set, inputs, transformer):
        self.computation = computation
        self.train_set = train_set
        self.inputs = inputs
        self.transformer = transformer

    def fill_feed_dict(self, dataset, feed_inputs):
        data = next(iter(dataset))
        return {feed_inputs[k]: data[k] for k in feed_inputs.keys()}

    def time(self, n_iterations, n_skip):
        """
        This runs _any_ computation repeatedly with data from feed_dict, and times it

        (Nothing model-specific inside, can be reused)
        """
        times = DefaultOrderedDict()
        feed_dict = self.fill_feed_dict(self.train_set, self.inputs)
        with closing(ngt.make_transformer_factory(self.transformer)()) as transformer:
            nviz = ngraph.transformers.passes.nviz.VizPass(show_axes=True,
                                                           show_all_metadata=False)
            transformer.register_graph_pass(nviz)
            model_out_computation = transformer.add_computation(self.computation)
            for i in range(n_skip):
                model_out_computation(feed_dict=feed_dict)
            for i in range(n_iterations):
                times[i]['start'] = time.time() * 1000.0
                model_out_computation(feed_dict=feed_dict)
                times[i]['stop'] = time.time() * 1000.0
        return times

    @staticmethod
    def print_benchmark_results(benchmarks):
        for label in benchmarks.keys():
            k = 0
            compute_time = np.zeros(len(benchmarks[label]))
            for v in benchmarks[label].values():
                compute_time[k] = v.values()[1] - v.values()[0]
                k += 1
            header = ('Func', 'Sum', 'Mean', 'Min', 'Max', 'Units')
            formatter = '| {:^20} ' * len(header) + '|'

            head_str = formatter.format(*header)
            sep = '-' * len(head_str)
            results = (label, compute_time.sum(), compute_time.mean(),
                       compute_time.min(), compute_time.max(), 'msec')
            results_str = formatter.format(*results)
            print(sep)
            print(head_str)
            print(sep)
            print(results_str)
            print(sep)
