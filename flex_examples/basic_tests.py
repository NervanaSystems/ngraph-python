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

parser = NgraphArgparser(description='x + 1.5, sum, abs')
args = parser.parse_args()
transformer_name = args.backend


def print_fm_stats(transformer):
    if transformer_name == 'flexgpu' and transformer.flex_manager.num_flex_tensors < 20:
        print("flex_manager.stat_ids after computations", transformer.flex_manager.stat_ids)
        fm = transformer.flex_manager

        fm.transfer_stats()
        print("flex_manager.host_stats", fm.host_stats)


# x + 1.5
x = ng.placeholder(())
x_plus_const = x + 1.5

transformer1 = ngt.make_transformer()
plus_const = transformer1.computation(x_plus_const, x)

for i in range(5):
    assert plus_const(i) == i + 1.5
    print(plus_const(i))

print_fm_stats(transformer1)

# absolute value of matrix
n, m = 2, 3
N = ng.make_axis(length=n)
M = ng.make_axis(length=m)
Zin = ng.placeholder((N, M))
Zout = abs(Zin)

transformer2 = ngt.make_transformer()
abs_func = transformer2.computation(Zout, Zin)

Xval = np.array([5, 1, 0, -2, 3, 4]).reshape(n, m).astype(np.float32)
Xval[0,1] = -Xval[0,1]
print(Xval)
print(abs_func(Xval))
assert np.allclose(abs_func(Xval), abs(Xval))

print_fm_stats(transformer2)

# sum
nelems = 10
H = ng.make_axis(length=nelems)
x2 = ng.placeholder(H)
z = ng.sum(x2)

transformer3 = ngt.make_transformer()
sum_func = transformer3.computation(z, x2)

xval = np.array(list(range(1,10) + [-1]))
xval[2] += + 1 + 10

print(sum(xval))
print(sum_func(xval))
assert(sum_func(xval) == sum(xval))

print_fm_stats(transformer3)

