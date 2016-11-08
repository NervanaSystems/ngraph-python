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
import ngraph as ng
import ngraph.transformers as ngt
import itertools as itt
import numpy as np
import copy

from ngraph.frontends.neon import GradientDescentMomentum
from ngraph.util.utils import ExecutorFactory

from neon.optimizers import GradientDescentMomentum as NeonGradientDescentMomentum
from neon import NervanaObject


def pytest_generate_tests(metafunc):
    if 'args' in metafunc.fixturenames:
        fargs = []
        lr = np.random.random(4)
        momentum = np.random.random(4)
        wdecay = [0.0005, 0.000, 0.001, 0.1]
        fargs = itt.product(lr, momentum, wdecay)
        metafunc.parametrize('args', fargs)


class DummyLayer(object):

    def __init__(self, p):
        self.p = p[0]

    def get_params(self):
        return self.p


def generate_data(C, N):
    w_init = np.random.rand(C.length).astype('float32')
    x = np.random.rand(C.length, N.length).astype('float32')
    y = np.random.rand(N.length).astype('float32')

    return x, y, w_init


def test_gdm(args, transformer_factory):
    """
    Test the ngraph GradientDescentMomentum against the neon version across 10 update steps.
    """
    # set up parameters
    C = ng.make_axis(name="C", length=200)
    N = ng.make_axis(name="N", length=128)

    # restrict to numpy transformer for now
    factory = ngt.make_transformer_factory('numpy')
    ngt.set_transformer_factory(factory)
    ngt.make_transformer()
    be = NervanaObject.be
    be.bsz = N.length

    # generate dummy data (to initialize values)
    (x, y, w_init) = generate_data(C, N)

    # set up nervana graph
    X = ng.placeholder([C, N], name='X')
    Y = ng.placeholder([N], name='Y')
    W = ng.variable([C], name='W', initial_value=w_init)

    ex = ExecutorFactory()
    transformer = ex.transformer

    lrate, mom, wdecay = args
    gdm = GradientDescentMomentum(learning_rate=lrate, momentum_coef=mom, wdecay=wdecay)
    cost = ng.sum(Y - ng.dot(W, X), out_axes=()) / N.length

    # to call ngraph gdm, use (ngraph_W, _) = ngraph_optimize(x, y)
    # where (x, y) are nparrays that fill the placeholders X and Y
    updates = gdm.configure(cost)
    ngraph_optimize = transformer.computation([W, updates], X, Y)
    transformer.initialize()
    gdm.optimize(epoch=0)

    # set up the neon gdm
    neon_gdm = NeonGradientDescentMomentum(learning_rate=lrate, momentum_coef=mom, wdecay=wdecay)
    dev_v0 = be.zeros((C.length, 1))  # velocities are zero at the beginning
    dev_dw = be.zeros((C.length, 1))  # we fill the gradient info in the below
    dev_w_init = be.array(w_init)  # copy w_init to device
    param_list = [((dev_w_init, dev_dw), [dev_v0])]

    # store the weights with each minibatch for debugging
    ng_Ws = []
    be_Ws = []
    for i in range(0, 20):  # run for 20 minibatches
        # obtain ngraph results
        (ng_W, _) = ngraph_optimize(x, y)
        ng_Ws.append(copy.deepcopy(ng_W))

        # obtain neon results
        dw = -1 * x.sum(axis=1)   # the gradients we compute analytically
        param_list[0][0][1].set(dw)  # fill the gradient

        neon_gdm.optimize([DummyLayer(param_list)], epoch=0)
        (param, grad), states = param_list[0]
        be_W = param.get()[:, 0]
        be_Ws.append(be_W)

        np.testing.assert_allclose(be_W, ng_W, rtol=1e-3)

        # generate dummy data for the next minibatch
        (x, y, _) = generate_data(C, N)
