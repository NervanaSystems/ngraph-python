from __future__ import division, print_function
import geon as ng
import gendata

N = 128
C = 4
g = gendata.MixtureGenerator([.5, .5], C)
XS, YS = g.gen_data(N, 10)
EVAL_XS, EVAL_YS = g.gen_data(N, 4)

X = ng.placeholder(axes=ng.Axes([C, N]))
Y = ng.placeholder(axes=ng.Axes([N]))
alpha = ng.placeholder(axes=ng.Axes())

W = ng.Variable(axes=ng.Axes([C]), initial_value=0)

Y_hat = ng.sigmoid(ng.dot(W, X))
L = ng.cross_entropy_binary(Y_hat, Y) / ng.tensor_size(Y_hat)

grad = ng.deriv(L, W)

update = ng.assign(W, W - alpha * grad)

transformer = ng.NumPyTransformer()
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
