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
from ngraph.frontends.neon import NgraphArgparser


# Select a transformer
parser = NgraphArgparser(description='simple gemm example')
args = parser.parse_args()
transformer_name = args.backend

factory = ngt.make_transformer_factory(transformer_name)
ngt.set_transformer_factory(factory)

print("\n--------- multiply -----------\n")

# matrix multiply

def test_gemm(n, c):
    ax = ng.make_name_scope("ax")
    ax.N = ng.make_axis(length=n, batch=True)
    ax.C = ng.make_axis(length=c)

    X = ng.placeholder(axes=[ax.C, ax.N])  # input 4x32
    Y = ng.placeholder(axes=[ax.N])  # output 1x32

    #W = ng.placeholder(axes=[ax.C - 1])  # "dual offsets" of +/- 1 to mark which axes should be matched (size 4x1)
    W = ng.variable(axes=[ax.C - 1], initial_value=0.1)  # "dual offsets" of +/- 1 to mark which axes should be matched (size 4x1)

    Y_hat = ng.dot(W, X)

    # computation
    transformer = ngt.make_transformer()
    update_fun = transformer.computation([Y_hat], X)

    w = np.ones(c)*0.1
    xs = np.ones(n*c).reshape(c, n)

    for ii in range(3):
        y_hat_val = update_fun(xs)
        print("Y: %s" % y_hat_val)

    return transformer

transformer = test_gemm(n=32, c=32)

if transformer_name == 'flexgpu' and transformer.flex_manager.num_flex_tensors < 20:
    print(transformer.flex_manager.stat_ids)
    fm = transformer.flex_manager

    print(fm.host_stats)
    fm.transfer_stats()
    print(fm.host_stats)
