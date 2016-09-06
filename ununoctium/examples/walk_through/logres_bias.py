from __future__ import division, print_function

import numpy as np
import geon

C = geon.Axis("C")
N = geon.Axis("N")

X = geon.placeholder(axes=geon.Axes([C, N]))
Y = geon.placeholder(axes=geon.Axes([N]))
alpha = geon.placeholder(axes=geon.Axes())

W = geon.Variable(axes=geon.Axes([C]), initial_value=0)
b = geon.Variable(axes=geon.Axes(), initial_value=0)

Y_hat = geon.sigmoid(geon.dot(W, X) + b)
L = geon.cross_entropy_binary(Y_hat, Y, out_axes=geon.Axes())

updates = [geon.assign(v, v - alpha * geon.deriv(L, v) / geon.tensor_size(Y_hat))
           for v in L.variables()]

all_updates = geon.doall(updates)

xs = np.array([[0.52, 1.12, 0.77],
               [0.88, -1.08, 0.15],
               [0.52, 0.06, -1.30],
               [0.74, -2.49, 1.39]]).T

ys = np.array([1, 1, 0, 1])

import gendata
g = gendata.MixtureGenerator([.5, .5], 10)
xs, ys = g.make_mixture(100)
g.fill_mixture(xs, ys)


C.length, N.length = xs.shape
transformer = geon.NumPyTransformer()
update_fun = transformer.computation([L, W, b, all_updates], alpha, X, Y)

for i in range(20):
    loss_val, w_val, b_val, _ = update_fun(5.0 / (1 + i), xs, ys)
    print("W: %s, b: %s, loss %s" % (w_val, b_val, loss_val))
