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
"""
To visualize HeTr computational graph with Tensorboard

1. run `python dist_hetr.py --graph_vis`

2. run `tensorboard --logdir /tmp/hetr_tb/ --port 6006`

use ssh port forwarding to run on remote server
https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server
"""
from __future__ import print_function
from contextlib import closing
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.neon import NgraphArgparser
import numpy as np

# Command Line Parser
parser = NgraphArgparser(description="Distributed HeTr Example")
parser.add_argument("--graph_vis", action="store_true", help="enable graph visualization")
args = parser.parse_args()

# Build the graph
H = ng.make_axis(length=6, name='height')
N = ng.make_axis(length=8, name='batch')
W1 = ng.make_axis(length=2, name='W1')
W2 = ng.make_axis(length=4, name='W2')
x = ng.placeholder(axes=[H, N])
w1 = ng.placeholder(axes=[W1, H])
w2 = ng.placeholder(axes=[W2, W1])
with ng.metadata(device_id=('0', '1'), parallel=N):
    dot1 = ng.dot(w1, x).named("dot1")
dot2 = ng.dot(w2, dot1).named("dot2")

np_x = np.random.randint(100, size=[H.length, N.length])
np_w1 = np.random.randint(100, size=[W1.length, H.length])
np_w2 = np.random.randint(100, size=[W2.length, W1.length])
with closing(ngt.make_transformer_factory('hetr', device='cpu')()) as transformer:
    computation = transformer.computation([dot1, dot2], x, w1, w2)
    res1, res2 = computation(np_x, np_w1, np_w2)
    print(res1, res2)
