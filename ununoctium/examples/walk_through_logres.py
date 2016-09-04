import numpy as np
import geon

xs = np.array([[0.52, 1.12, 0.77],
               [0.88, -1.08, 0.15],
               [0.52, 0.06, -1.30],
               [0.74, -2.49, 1.39]]).T

ys = np.array([1, 1, 0, 1])

C, N = xs.shape

X = geon.placeholder(axes=geon.Axes([C, N]))
Y = geon.placeholder(axes=geon.Axes([N]))
alpha = geon.placeholder(axes=geon.Axes())

W = geon.Variable(axes=geon.Axes([C]), initial_value=0)

Y_hat = geon.sigmoid(geon.dot(W, X))
L = geon.cross_entropy_binary(Y_hat, Y)

grad = geon.deriv(L, W)

update = geon.assign(W, W - alpha * grad)

transformer = geon.NumPyTransformer()
update_fun = transformer.computation([L, W, update], alpha, X, Y)

for i in range(10):
    loss_val, w_val, _ = update_fun(.1, xs, ys)
    print("W: %s, loss %s" % (w_val, loss_val))
