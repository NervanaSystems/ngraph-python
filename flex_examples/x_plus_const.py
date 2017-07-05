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
import numpy as np
import ngraph as ng
import ngraph.transformers as ngt
import ngraph.op_graph.axes as ax
from ngraph.frontends.neon.argparser import NgraphArgparser

parser = NgraphArgparser(description='x + 1.5 example')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=5)
args = parser.parse_args()
start = args.start
end = args.end

transformer = ngt.make_transformer()

# Build the graph
x = ng.placeholder(())
x_plus_one = x + 1.5

# Define a computation
plus_one = transformer.computation(x_plus_one, x)

# Run the computation
for i in range(start, end):
    print(plus_one(i))
    print()
