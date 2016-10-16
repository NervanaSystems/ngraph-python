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
import ngraph
import itertools as itt
import numpy as np
import copy

from ngraph.frontends.neon import GradientDescentMomentum

from neon import NervanaObject
from neon.optimizers import GradientDescentMomentum as NeonGradientDescentMomentum
from neon.backends import gen_backend
import pytest


def pytest_generate_tests(metafunc):
    if 'args' in metafunc.fixturenames:
        fargs = []
        lr = np.random.random(4)
        momentum = np.random.random(4)
        wdecay = [0.0005, 0.000, 0.001, 0.1]
        fargs = itt.product(lr, momentum, wdecay)
        metafunc.parametrize('args', fargs)


def wrap(x):
    be = NervanaObject.be
    dtypeu = np.float32
    return be.array(dtypeu(x))


class DummyLayer(object):

    def __init__(self, p):
        self.p = p[0]

    def get_params(self):
        return self.p


def generate_data(C, N):
    w_init = np.random.rand(C).astype('float32')
    x = np.random.rand(C, N).astype('float32')
    y = np.random.rand(N).astype('float32')

    return x, y, w_init


# xfail due to initial_value=nparray not working
# this test was working a previous commit of ngraph
@pytest.mark.xfail
def test_gdm(args):
    """
    Test the ngraph GradientDescentMomentum against the neon version across 10 update steps.

    This test currently fails. Uncommenting one of the lines (marked below) will cause the test
    to pass on the first minibatch, but it will fail subsequently.
    """
    # set up parameters
    C = ngraph.Axis("C")
    N = ngraph.Axis("N")

    C.length = 200
    N.length = 128

    be = gen_backend(backend='cpu', batch_size=128)

    # generate dummy data (to initialize values)
    (x, y, w_init) = generate_data(C.length, N.length)

    # set up nervana graph
    X = ngraph.placeholder(axes=ngraph.Axes([C, N]), name='X')
    Y = ngraph.placeholder(axes=ngraph.Axes([N]), name='Y')
    W = ngraph.Variable(axes=ngraph.Axes([C]), initial_value=w_init)

    transformer = ngraph.NumPyTransformer()

    lrate, mom, wdecay = args
    gdm = GradientDescentMomentum(learning_rate=lrate, momentum_coef=mom, wdecay=wdecay)
    cost = ngraph.sum(Y - ngraph.dot(W, X), out_axis=())

    # to call ngraph gdm, use (ngraph_W, _) = ngraph_optimize(x, y)
    # where (x, y) are nparrays that fill the placeholders X and Y
    updates = gdm.configure(transformer, cost, N)
    ngraph_optimize = transformer.computation([W, updates], X, Y)
    transformer.initialize()
    gdm.optimize(epoch=0)

    # set up the neon gdm
    neon_gdm = NeonGradientDescentMomentum(learning_rate=lrate, momentum_coef=mom, wdecay=wdecay)
    dev_v0 = be.zeros((C, 1))  # velocities are zero at the beginning
    dev_dw = be.zeros((C, 1))  # we fill the gradient info in the below
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

        np.testing.assert_allclose(be_W, ng_W, rtol=1e-4)

        # generate dummy data for the next minibatch
        (x, y, _) = generate_data(C, N)


if __name__ == '__main__':
    be = gen_backend(backend='cpu', batch_size=128)
    test_gdm((1.0, 0.5, 0.0005))
