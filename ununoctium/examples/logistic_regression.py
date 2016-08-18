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

from __future__ import print_function
import numpy as np
import geon as be
import geon.frontends.base.axis as ax


def numpy_logistic_regression(xs, ys, max_iter, alpha):
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def predict(thetas, xs):
        return sigmoid(np.dot(xs, thetas))

    def get_loss(thetas, xs, ys):
        ys_pred = predict(thetas, xs)
        log_likelihoods = np.log(ys_pred) * ys + np.log(1 - ys_pred) * (1 - ys)
        loss = -np.sum(log_likelihoods)
        return loss

    def get_grad(thetas, xs, ys):
        ys_pred = predict(thetas, xs)
        grad = -np.dot(ys - ys_pred, xs)
        return grad

    # init weights
    thetas = np.array([0.0, 0.0, 0.0])

    # gradient descent
    for i in range(max_iter):
        # foward
        loss = get_loss(thetas, xs, ys)
        # backward
        grad = get_grad(thetas, xs, ys)
        # print
        print("grad: %s, loss %s" % (grad, loss))
        # update
        thetas -= grad * alpha

    return thetas


def geon_logistic_regression(xs_np, ys_np, max_iter, alpha):
    def sigmoid(x):
        # return 1. / (1. + be.exp(-x))
        return be.sigmoid(x)

    def predict(thetas, xs):
        return sigmoid(be.dot(xs, thetas))

    def get_loss(thetas, xs, ys):
        ys_pred = predict(thetas, xs)
        log_likelihoods = be.log(ys_pred) * ys + be.log(1 - ys_pred) * (1 - ys)
        loss = -be.sum(log_likelihoods)
        return loss

    with be.bound_environment():
        # axis
        ax.C.length = 3
        ax.Y.length = 1
        ax.N.length = 4

        # input tensors
        xs = be.placeholder(axes=(ax.C, ax.N))
        ys = be.placeholder(axes=(ax.Y, ax.N))
        xs.value = xs_np.transpose()
        ys.value = ys_np.reshape((ax.Y.length, ax.N.length))

        # init weights
        thetas_np = np.array([0., 0., 0.])
        thetas_numpy_tensor = be.NumPyTensor(thetas_np, axes=(ax.C,))
        thetas = be.Variable(initial_value=thetas_numpy_tensor, axes=(ax.C))

        # computations
        loss = get_loss(thetas, xs, ys)

        # auto-diff
        variable = list(loss.variables())[0]  # we only have one variable
        grad = be.deriv(loss, variable)

        # update rule
        update = be.assign(lvalue=variable, rvalue=variable - alpha * grad)

        # transformer
        transformer = be.NumPyTransformer()
        train_eval_comp = transformer.computation([grad, update, loss])
        transformer.finalize()

        # evaluate
        for i in range(max_iter):
            result_dict = train_eval_comp.evaluate()
            print("grad: %s, loss %s" % (result_dict[grad], result_dict[loss]))


if __name__ == '__main__':
    # setups
    xs = np.array([[0.52, 1.12, 0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
    ys = np.array([1, 1, 0, 1])
    max_iter = 10
    alpha = 0.1

    # numpy
    print("# numpy training")
    numpy_logistic_regression(xs, ys, max_iter, alpha)

    # geon
    print("# geon training")
    geon_logistic_regression(xs, ys, max_iter, alpha)
