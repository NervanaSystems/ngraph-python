from __future__ import division, print_function

import ngraph as ng
import ngraph.transformers as ngt
import gendata

ax = ng.NameScope(name="ax")

ax.W = ng.make_axis()
ax.H = ng.make_axis()
ax.N = ng.make_axis()

X = ng.placeholder(axes=ng.make_axes([ax.W, ax.H, ax.N]))
Y = ng.placeholder(axes=ng.make_axes([ax.N]))
alpha = ng.placeholder(axes=ng.make_axes())

W = ng.Variable(axes=ng.make_axes([ax.W, ax.H]), initial_value=0)
b = ng.Variable(axes=ng.make_axes(), initial_value=0)

Y_hat = ng.sigmoid(ng.dot(W, X) + b)
L = ng.cross_entropy_binary(Y_hat, Y) / ng.tensor_size(Y_hat)

updates = [ng.assign(v, v - alpha * ng.deriv(L, v) / ng.tensor_size(Y_hat))
           for v in L.variables()]

all_updates = ng.doall(updates)

ax.W.length = 4
ax.H.length = 1
ax.N.length = 128

g = gendata.MixtureGenerator([.5, .5], (ax.W.length, ax.H.length))
XS, YS = g.gen_data(ax.N.length, 10)
EVAL_XS, EVAL_YS = g.gen_data(ax.N.length, 4)

transformer = ngt.make_transformer()

update_fun = transformer.computation([L, W, b, all_updates], alpha, X, Y)
eval_fun = transformer.computation(L, X, Y)


def avg_loss():
    total_loss = 0
    for xs, ys in zip(EVAL_XS, EVAL_YS):
        loss_val = eval_fun(xs, ys)
        total_loss += loss_val
    return total_loss / len(xs)

print("Starting avg loss: {}".format(avg_loss()))
for i in range(10):
    for xs, ys in zip(XS, YS):
        loss_val, w_val, b_val, _ = update_fun(5.0 / (1 + i), xs, ys)
    print("After epoch %d: W: %s, b: %s, avg loss %s" % (i, w_val.T, b_val, avg_loss()))
