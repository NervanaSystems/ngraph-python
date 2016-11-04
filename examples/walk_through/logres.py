from __future__ import division, print_function

import ngraph as ng
import ngraph.transformers as ngt
import gendata

N = 128
C = 4
g = gendata.MixtureGenerator([.5, .5], C)
XS, YS = g.gen_data(N, 10)

X = ng.placeholder(axes=ng.make_axes([C, N]))
Y = ng.placeholder(axes=ng.make_axes([N]))
alpha = ng.placeholder(axes=ng.make_axes())

W = ng.Variable(axes=ng.make_axes([C]), initial_value=0)

Y_hat = ng.sigmoid(ng.dot(W, X))
L = ng.cross_entropy_binary(Y_hat, Y) / ng.tensor_size(Y_hat)

grad = ng.deriv(L, W)

update = ng.assign(W, W - alpha * grad)

transformer = ngt.make_transformer()
update_fun = transformer.computation([L, W, update], alpha, X, Y)

for i in range(10):
    for xs, ys in zip(XS, YS):
        loss_val, w_val, _ = update_fun(5.0 / (1 + i), xs, ys)
        print("W: %s, loss %s" % (w_val, loss_val))
