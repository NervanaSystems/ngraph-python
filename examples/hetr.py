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
from __future__ import print_function
import ngraph as ng
import ngraph.transformers as ngt
import ngraph.transformers.passes.nviz

# Build the graph
with ng.metadata(device='numpy1'):
    x = ng.placeholder(())
x_plus_one = x + 1

# Select a transformer
transformer = ngt.make_transformer_factory('hetr')()

# visualize the graph
transformer.register_graph_pass(ngraph.transformers.passes.nviz.VizPass(show_all_metadata=True))
#import ipdb; ipdb.set_trace()
# Define a computation
plus_one = transformer.computation(x_plus_one, x)

# Run the computation
for i in range(3):
    print(plus_one(i))

