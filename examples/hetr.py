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
from contextlib import closing
import ngraph as ng
import ngraph.transformers as ngt
import ngraph.transformers.passes.nviz
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iter_count", "-i", type=int, default=5, help="num iterations to run")
parser.add_argument("--visualize", "-v", action="store_true", help="enable graph visualization")
args = parser.parse_args()

# Build the graph
with ng.metadata(device_id='1'):
    x = ng.placeholder(())
x_plus_one = x + 1

# Select a transformer
with closing(ngt.make_transformer_factory('hetr')()) as hetr:

    # Visualize the graph
    if args.visualize:
        hetr.register_graph_pass(ngraph.transformers.passes.nviz.VizPass(show_all_metadata=True))

    # Define a computation
    plus_one = hetr.computation(x_plus_one, x)

    # Run the computation
    for i in range(args.iter_count):
        print(plus_one(i))
