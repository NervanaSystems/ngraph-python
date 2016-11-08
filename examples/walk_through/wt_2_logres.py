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
from __future__ import division, print_function

import ngraph as ng
import ngraph.transformers as ngt
import gendata

ax = ng.make_name_scope("ax")
ax.N = ng.make_axis(length=128)
ax.C = ng.make_axis(length=4)

g = gendata.MixtureGenerator([.5, .5], (ax.C.length,))
XS, YS = g.gen_data(ax.N.length, 10)

alpha = ng.placeholder(())
X = ng.placeholder([ax.C, ax.N])
Y = ng.placeholder([ax.N])

W = ng.variable([ax.C - 1], initial_value=0)

Y_hat = ng.sigmoid(ng.dot(W, X, use_dual=True))
L = ng.cross_entropy_binary(Y_hat, Y) / ng.tensor_size(Y_hat)

grad = ng.deriv(L, W)

update = ng.assign(W, W - alpha * grad)

transformer = ngt.make_transformer()
update_fun = transformer.computation([L, W, update], alpha, X, Y)

for i in range(10):
    for xs, ys in zip(XS, YS):
        loss_val, w_val, _ = update_fun(5.0 / (1 + i), xs, ys)
        print("W: %s, loss %s" % (w_val, loss_val))
