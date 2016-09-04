import numpy as np
import geon

C = geon.Axis("C")
N = geon.Axis("N")

X = geon.placeholder(axes=geon.Axes([C, N]))
Y = geon.placeholder(axes=geon.Axes([N]))
alpha = geon.placeholder(axes=geon.Axes())

W = geon.Variable(axes=geon.Axes([C]), initial_value=0)

Y_hat = geon.sigmoid(geon.dot(W, X))
L = geon.cross_entropy_binary(Y_hat, Y, out_axes=geon.Axes())

grad = geon.deriv(L, W)

update = geon.assign(W, W - alpha * grad / geon.tensor_size(Y_hat))

xs = np.array([[0.52, 1.12, 0.77],
               [0.88, -1.08, 0.15],
               [0.52, 0.06, -1.30],
               [0.74, -2.49, 1.39]]).T

ys = np.array([1, 1, 0, 1])

C.length, N.length = xs.shape
transformer = geon.NumPyTransformer()
update_fun = transformer.computation([L, W, update], alpha, X, Y)

for i in range(20):
    loss_val, w_val, _ = update_fun(5.0 / (1 + i), xs, ys)
    print("W: %s, loss %s" % (w_val, loss_val))
