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

ax = ng.make_name_scope(name="ax")

ax.W = ng.make_axis(length=1)
ax.H = ng.make_axis(length=4)
ax.N = ng.make_axis(length=128, batch=True)

g = gendata.MixtureGenerator([.5, .5], (ax.W.length, ax.H.length))
XS, YS = g.gen_data(ax.N.length, 10)

EVAL_XS, EVAL_YS = g.gen_data(ax.N.length, 4)

alpha = ng.placeholder(())
X = ng.placeholder([ax.W, ax.H, ax.N])
Y = ng.placeholder([ax.N])

W = ng.variable([ax.W - 1, ax.H - 1], initial_value=0)
b = ng.variable((), initial_value=0)

Y_hat = ng.sigmoid(ng.dot(W, X, name="WX") + b)
L = ng.cross_entropy_binary(Y_hat, Y, out_axes=None) / ng.batch_size(Y_hat)

updates = [ng.assign(v, v - alpha * ng.deriv(L, v) / ng.batch_size(Y_hat))
           for v in L.variables()]

all_updates = ng.doall(updates)


transformer = ngt.make_transformer()

update_fun = transformer.computation([L, W, b, all_updates], alpha, X, Y)
eval_fun = transformer.computation(L, X, Y)


def avg_loss():
    total_loss = 0
    for xs, ys in zip(EVAL_XS, EVAL_YS):
        loss_val = eval_fun(xs, ys)
        total_loss += loss_val
    return total_loss / xs.shape[-1]

print("Starting avg loss: {}".format(avg_loss()))
for i in range(10):
    for xs, ys in zip(XS, YS):
        loss_val, w_val, b_val, _ = update_fun(5.0 / (1 + i), xs, ys)
    print("After epoch %d: W: %s, b: %s, avg loss %s" % (i, w_val.T, b_val, avg_loss()))
