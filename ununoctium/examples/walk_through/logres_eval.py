from __future__ import division, print_function
import geon
import gendata

N = 128
C = 4
g = gendata.MixtureGenerator([.5, .5], C)
XS, YS = g.gen_data(N, 10)
EVAL_XS, EVAL_YS = g.gen_data(N, 4)

X = geon.placeholder(axes=geon.Axes([C, N]))
Y = geon.placeholder(axes=geon.Axes([N]))
alpha = geon.placeholder(axes=geon.Axes())

W = geon.Variable(axes=geon.Axes([C]), initial_value=0)

Y_hat = geon.sigmoid(geon.dot(W, X))
L = geon.cross_entropy_binary(Y_hat, Y) / geon.tensor_size(Y_hat)

grad = geon.deriv(L, W)

update = geon.assign(W, W - alpha * grad)

transformer = geon.NumPyTransformer()
update_fun = transformer.computation([L, W, update], alpha, X, Y)
eval_fun = transformer.computation(L, X, Y)

for i in range(10):
    for xs, ys in zip(XS, YS):
        loss_val, w_val, _ = update_fun(5.0 / (1 + i), xs, ys)
        print("W: %s, loss %s" % (w_val, loss_val))

total_loss = 0
for xs, ys in zip(EVAL_XS, EVAL_YS):
    loss_val = eval_fun(xs, ys)
    total_loss += loss_val
print("Loss: {}".format(total_loss / len(xs)))
