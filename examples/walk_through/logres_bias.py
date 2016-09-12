from __future__ import division, print_function

import ngraph as ng
import gendata

C = ng.Axis("C")
N = ng.Axis("N")

X = ng.placeholder(axes=ng.Axes([C, N]))
Y = ng.placeholder(axes=ng.Axes([N]))
alpha = ng.placeholder(axes=ng.Axes())

W = ng.Variable(axes=ng.Axes([C]), initial_value=0)
b = ng.Variable(axes=ng.Axes(), initial_value=0)

Y_hat = ng.sigmoid(ng.dot(W, X) + b)
L = ng.cross_entropy_binary(Y_hat, Y) / ng.tensor_size(Y_hat)

updates = [ng.assign(v, v - alpha * ng.deriv(L, v))
           for v in L.variables()]

all_updates = ng.doall(updates)

C.length = 4
N.length = 128

g = gendata.MixtureGenerator([.5, .5], C.length)
XS, YS = g.gen_data(N.length, 10)
EVAL_XS, EVAL_YS = g.gen_data(N.length, 4)

transformer = ng.NumPyTransformer()

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
    print("After epoch %d: W: %s, b: %s, avg loss %s" % (i, w_val, b_val, avg_loss()))
