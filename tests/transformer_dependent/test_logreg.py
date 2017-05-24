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
from ngraph.testing import ExecutorFactory
import pytest

pytestmark = pytest.mark.transformer_dependent("module")


class NumpyLogreg(object):
    def __init__(self, xs, ys, thetas):
        self.xs = xs.T.copy()
        self.ys = ys
        self.thetas = thetas.copy()

    def optimize(self, alpha):
        def sigmoid(x):
            return 1. / (1. + np.exp(-x))

        ys_pred = sigmoid(np.dot(self.xs, self.thetas))
        log_likelihoods = np.log(ys_pred) * self.ys + np.log(1 - ys_pred) * (1 - self.ys)
        loss = -np.sum(log_likelihoods)
        grad = -np.dot(self.ys - ys_pred, self.xs)
        self.thetas -= grad * alpha
        return grad, loss, self.thetas


def test_logreg(transformer_factory):
    # xs: (C, N), y: (N,)
    xs = np.array([[0.52, 0.88, 0.52, 0.74],
                   [1.12, -1.08, 0.06, -2.49],
                   [0.77, 0.15, -1.3, 1.39]])
    ys = np.array([1, 1, 0, 1])
    max_iter = 10
    alpha = 0.1
    thetas = np.array([0., 0., 0.])

    np_logreg = NumpyLogreg(xs, ys, thetas)

    C, N = ng.make_axis(length=3), ng.make_axis(length=4)

    # input tensors
    xs_v = ng.placeholder((C, N))
    ys_v = ng.placeholder([N])
    alpha_v = ng.placeholder(())
    thetas_var = ng.variable([C], initial_value=thetas)

    # define ops
    ys_pred = ng.sigmoid(ng.dot(thetas_var, xs_v))
    log_likelihoods = ng.log(ys_pred) * ys_v + ng.log(1 - ys_pred) * (1 - ys_v)
    loss = -ng.sum(log_likelihoods, reduction_axes=[N])
    grad_comp = ng.deriv(loss, thetas_var)
    grad = ng.sequential([
        ng.assign(thetas_var, thetas_var - alpha_v * grad_comp),
        thetas_var,
        grad_comp
    ])

    # transformer
    with ExecutorFactory() as ex:
        train_eval_func = ex.executor([grad, loss, thetas_var],
                                      xs_v, ys_v, alpha_v)

        # evaluate
        for i in range(max_iter):
            grad_np, loss_np, thetas_np = np_logreg.optimize(alpha)
            grad_ng, loss_ng, thetas_ng = train_eval_func(xs, ys, alpha)
            assert ng.testing.allclose(loss_np, loss_ng)
            assert ng.testing.allclose(grad_np, grad_ng)
            assert ng.testing.allclose(thetas_np, thetas_ng)
