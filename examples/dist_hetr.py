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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iter_count", "-i", type=int, default=5, help="num iterations to run")
parser.add_argument("--visualize", "-v", action="store_true", help="enable graph visualization")
args = parser.parse_args()

# Build the graph
H = ng.make_axis(length=4, name='height')
W = ng.make_axis(length=6, name='width')

x = ng.placeholder(axes=[H, W])
with ng.metadata(device_id=('1', '2'), parallel=W):
    x_plus_one = x + 1

x_plus_two = x_plus_one + 1

# Select a transformer
hetr = ngt.make_transformer_factory('hetr')()

# Visualize the graph
if args.visualize:
    hetr.vizpass = ngraph.transformers.passes.nviz.VizPass(show_all_metadata=True, show_axes=True)

# Define a computation
plus_two = hetr.computation(x_plus_two, x)

# Run the computation
for i in range(args.iter_count):
    print(plus_two(i))

hetr.cleanup()
