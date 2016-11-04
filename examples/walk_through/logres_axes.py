from __future__ import division, print_function

import ngraph as ng
import ngraph.transformers as ngt
import gendata

C = ng.make_axis("C")
N = ng.make_axis("N")

X = ng.placeholder(axes=ng.make_axes([C, N]))
Y = ng.placeholder(axes=ng.make_axes([N]))
alpha = ng.placeholder(axes=ng.make_axes())

W = ng.Variable(axes=ng.make_axes([C]), initial_value=0)

Y_hat = ng.sigmoid(ng.dot(W, X))
L = ng.cross_entropy_binary(Y_hat, Y) / ng.tensor_size(Y_hat)

grad = ng.deriv(L, W)

update = ng.assign(W, W - alpha * grad)

C.length = 4
N.length = 128

g = gendata.MixtureGenerator([.5, .5], C.length)
XS, YS = g.gen_data(N.length, 10)
EVAL_XS, EVAL_YS = g.gen_data(N.length, 4)

transformer = ngt.make_transformer()

update_fun = transformer.computation([L, W, update], alpha, X, Y)
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
        loss_val, w_val, _ = update_fun(5.0 / (1 + i), xs, ys)
    print("After epoch %d: W: %s, avg loss %s" % (i, w_val, avg_loss()))
