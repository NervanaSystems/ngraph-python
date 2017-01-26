# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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

'''
Test of the optimizers
'''
import copy
import itertools as itt
import numpy as np

import ngraph as ng
from ngraph.frontends.neon import GradientDescentMomentum
from ngraph.testing.execution import ExecutorFactory


def pytest_generate_tests(metafunc):
    if 'args' in metafunc.fixturenames:
        lr = np.random.random(2)
        momentum = np.random.random(4)
        wdecay = [0.0005, 0.000, 0.001, 0.1]
        fargs = itt.product(lr, momentum, wdecay)
        metafunc.parametrize('args', fargs)


def generate_data(C, N):
    x = np.random.rand(C, N).astype('float32')
    y = np.random.rand(N).astype('float32')

    return x, y


def compare_optimizers(ng_opt, np_opt, nfeatures=20, batch_size=32, niters=20, rtol=1e-6):

    ng_Ws = list()
    np_Ws = list()
    for x, y in (generate_data(nfeatures, batch_size) for _ in range(niters)):
        # NOTE: Does not update learning rate according to schedule.
        # NOTE: Keep in mind for schedule tests
        # Compute ngraph values
        ng_W, _ = ng_opt(x, y)
        ng_Ws.append(copy.deepcopy(ng_W))

        # Compute numpy values
        np_W = np_opt(x, y)
        np_Ws.append(copy.deepcopy(np_W))

        ng.testing.assert_allclose(np_W, ng_W, rtol=rtol)


# xfail due to initial_value=nparray not working
# this test was working a previous commit of ngraph
# @pytest.mark.xfail(strict=True)
def test_gdm(args, transformer_factory):
    """
    Test the ngraph GradientDescentMomentum against the neon version across 10 update steps.
    """
    # set up parameters
    C = ng.make_axis(20, name="C")
    N = ng.make_axis(32, name="N", batch=True)

    # generate dummy data (to initialize values)
    w_init = np.random.rand(C.length).astype('float32')

    # set up nervana graph
    X = ng.placeholder([C, N]).named('X')
    Y = ng.placeholder([N]).named('Y')
    W = ng.variable([C - 1], initial_value=w_init).named('W')

    ex = ExecutorFactory()
    transformer = ex.transformer

    lrate, mom, wdecay = args
    gdm = GradientDescentMomentum(learning_rate=lrate, momentum_coef=mom, wdecay=wdecay)
    cost = ng.sum(Y - ng.dot(W, X), out_axis=())

    # to call ngraph gdm, use (ngraph_W, _) = ngraph_optimize(x, y)
    # where (x, y) are nparrays that fill the placeholders X and Y
    updates = gdm(cost)
    ngraph_optimize = transformer.computation([W, updates], X, Y)

    # set up the reference values for gradient descent
    w_ref = w_init.copy()
    vel_ref = np.zeros_like(w_ref)

    # store the weights with each minibatch for debugging
    ng_Ws = []

    # run for 20 minibatches
    for i, (x, y) in enumerate([generate_data(C.length, N.length) for _ in range(20)]):
        # obtain ngraph results
        (ng_W, _) = ngraph_optimize(x, y)
        gdm.update_learning_rate()
        ng_Ws.append(copy.deepcopy(ng_W))

        # obtain reference results
        dw = -1 * x.sum(axis=1) / N.length   # the gradients we compute analytically

        dw = dw + wdecay * w_ref
        if mom == 0:
            w_ref[:] = w_ref - lrate * dw
        else:
            vel_ref[:] = mom * vel_ref - lrate * dw
            w_ref[:] = w_ref + vel_ref

        ng.testing.assert_allclose(w_ref, ng_W, rtol=1e-3)


def test_gdm_nesterov(args, transformer_factory):
    """
    Test ngraph gradient descent with nesterov momentum against a simple numpy implementation
    Args:
        args (tuple): learning rate, momentum, weight decay
        transformer_factory: unused
    """

    lrate, mom, wdecay = args
    ex = ExecutorFactory()
    transformer = ex.transformer

    gdm = GradientDescentMomentum(learning_rate=lrate, momentum_coef=mom,
                                  wdecay=wdecay, nesterov=True)

    C = ng.make_axis(20, name="C")
    N = ng.make_axis(32, name="N", batch=True)

    # params to be updated using GDM
    np_W = np.random.rand(C.length)
    W = ng.variable([C - 1], initial_value=np_W)

    # Set up initial velocity
    velocity = np.zeros(C.length)

    # Set up data placeholders
    data = ng.placeholder([C, N]).named("data")
    target = ng.placeholder([N]).named("target")

    # Set up op graph
    cost = ng.sum(target - ng.dot(W, data), out_axis=())
    updates = gdm(cost)
    optimize = transformer.computation([W, updates], data, target)

    # Set up numpy version
    def numpy_nesterov(x, y):
        grad = -1 * x.mean(axis=1)
        velocity[:] = mom * velocity - lrate * (grad + wdecay * np_W)
        np_W[:] = np_W + mom * velocity - lrate * (grad + wdecay * np_W)

        return np_W

    compare_optimizers(optimize, numpy_nesterov, nfeatures=C.length, batch_size=N.length)
