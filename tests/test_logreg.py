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
import ngraph as ng
import ngraph.transformers as ngt


def numpy_logreg(xs, ys, max_iter, alpha):
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

    # convert to (N, C) layout
    xs = xs.T.copy()

    # init weights
    thetas = np.array([0.0, 0.0, 0.0])

    # gradient descent
    loss_collect = []
    grad_collect = []
    thetas_collect = []
    for i in range(max_iter):
        loss = get_loss(thetas, xs, ys)
        grad = get_grad(thetas, xs, ys)
        thetas -= grad * alpha
        loss_collect.append(loss)
        grad_collect.append(grad.copy())
        thetas_collect.append(thetas.copy())

    return loss_collect, grad_collect, thetas_collect


def ngraph_logreg(xs_np, ys_np, max_iter, alpha):
    # axis
    C, N = ng.make_axis("C"), ng.make_axis("N")

    def sigmoid(x):
        return 1. / (1. + ng.exp(-x))

    def predict(thetas, xs):
        return sigmoid(ng.dot(thetas, xs))

    def get_loss(thetas, xs, ys):
        ys_pred = predict(thetas, xs)
        log_likelihoods = ng.log(ys_pred) * ys + ng.log(1 - ys_pred) * (1 - ys)
        loss = -ng.sum(log_likelihoods, reduction_axes=[N])
        return loss

    # axis
    C.length = 3
    N.length = 4

    # input tensors
    xs = ng.placeholder((C, N))
    ys = ng.placeholder([N])

    # init weights
    thetas_np = np.array([0., 0., 0.])
    thetas_numpy_tensor = ng.constant(thetas_np, [C])
    thetas = ng.variable([C - 1], initial_value=thetas_numpy_tensor)

    # define ops
    loss = get_loss(thetas, xs, ys)
    variable = list(loss.variables())[0]  # we only have one variable thetas
    grad = ng.deriv(loss, variable)
    with ng.Op.saved_user_deps():
        update = ng.assign(variable, variable - alpha * grad)

    # transformer
    transformer = ngt.make_transformer()
    train_eval_func = transformer.computation([grad, loss, thetas, update],
                                              xs, ys)

    # evaluate
    loss_collect = []
    grad_collect = []
    thetas_collect = []
    for i in range(max_iter):
        grad_val, loss_val, thetas_val, _ = train_eval_func(xs_np, ys_np)
        loss_collect.append(loss_val.copy())
        grad_collect.append(grad_val.copy())
        thetas_collect.append(thetas_val.copy())

    return loss_collect, grad_collect, thetas_collect


def test_logreg(transformer_factory):
    # xs: (C, N), y: (N,)
    xs = np.array([[0.52, 0.88, 0.52, 0.74],
                   [1.12, -1.08, 0.06, -2.49],
                   [0.77, 0.15, -1.3, 1.39]])
    ys = np.array([1, 1, 0, 1])
    max_iter = 10
    alpha = 0.1

    # numpy
    np_loss, np_grad, np_thetas = numpy_logreg(xs, ys, max_iter, alpha)

    # ngraph
    ng_loss, ng_grad, ng_thetas = ngraph_logreg(xs, ys, max_iter, alpha)

    # asserts
    assert ng.testing.allclose(np_loss, ng_loss)
    assert ng.testing.allclose(np.asarray(np_grad), np.asarray(ng_grad))
    assert ng.testing.allclose(np.asarray(np_thetas), np.asarray(ng_thetas))
