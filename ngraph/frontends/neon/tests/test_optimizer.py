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
import itertools as itt
import numpy as np
import copy

from ngraph.frontends.neon import GradientDescentMomentum
from neon.optimizers import GradientDescentMomentum as NeonGradientDescentMomentum
from ngraph.transformers import Transformer
from neon import NervanaObject


def pytest_generate_tests(metafunc):
    if 'args' in metafunc.fixturenames:
        fargs = []
        # lr = np.random.random(2)
        lr = [0.1, 0.2]
        momentum = [0]
        wdecay = [0]
        # momentum = np.random.random(4)
        # wdecay = [0.0005, 0.000, 0.001, 0.1]
        fargs = itt.product(lr, momentum, wdecay)
        metafunc.parametrize('args', fargs)


class DummyLayer(object):

    def __init__(self, p):
        self.p = p[0]

    def get_params(self):
        return self.p


def generate_data(C, N):
    x = np.random.rand(C, N).astype('float32')
    y = np.random.rand(N).astype('float32')

    return x, y

# xfail due to initial_value=nparray not working
# this test was working a previous commit of ngraph
# @pytest.mark.xfail(strict=True)
def test_gdm(args, transformer_factory):
    """
    Test the ngraph GradientDescentMomentum against the neon version across 10 update steps.
    """
    # set up parameters
    # user_deps = ng.Op._get_thread_user_deps()
    # if len(user_deps)>1:
    #     import pdb; pdb.set_trace()
    #     user_deps.pop()

    C = ng.Axis(20, name="C")
    N = ng.Axis(32, name="N", batch=True)

    be = gen_backend(backend='cpu', batch_size=N.length)

    # generate dummy data (to initialize values)
    w_init = np.random.rand(C.length).astype('float32')

    # set up nervana graph
    X = ng.placeholder(axes=ng.Axes([C, N]), name='X')
    Y = ng.placeholder(axes=ng.Axes([N]), name='Y')
    I = ng.placeholder(axes=(), name='I')

    W = ng.Variable(axes=ng.Axes([C]), initial_value=w_init)


    transformer = ng.NumPyTransformer()

    lrate, mom, wdecay = args
    gdm = GradientDescentMomentum(learning_rate=lrate, momentum_coef=mom, wdecay=wdecay)
    cost = ng.sum(Y - ng.dot(W, X), out_axis=())

    # to call ngraph gdm, use (ngraph_W, _) = ngraph_optimize(x, y)
    # where (x, y) are nparrays that fill the placeholders X and Y
    updates = gdm(cost, I)
    ngraph_optimize = transformer.computation([W, updates], X, Y, I)
    wgetter = transformer.computation(W)
    transformer.initialize()

    # set up the neon gdm
    neon_gdm = NeonGradientDescentMomentum(learning_rate=lrate, momentum_coef=mom, wdecay=wdecay)
    # dev_v0 = be.zeros((C.length, 1))  # velocities are zero at the beginning
    dev_dw = be.zeros((C.length, 1))  # we fill the gradient info in the below
    dev_w_init = be.array(w_init)  # copy w_init to device
    param_list = [((dev_w_init, dev_dw), [])]

    print('')
    print(dev_w_init.get()[:,0])
    print(wgetter())
    # store the weights with each minibatch for debugging
    ng_Ws = []
    be_Ws = []

    # run for 20 minibatches
    for i, (x, y) in enumerate([generate_data(C.length, N.length) for _ in range(20)]):
        # obtain ngraph results
        (ng_W, _) = ngraph_optimize(x, y, i)
        ng_Ws.append(copy.deepcopy(ng_W))

        # obtain neon results
        dw = -1 * x.sum(axis=1)   # the gradients we compute analytically
        param_list[0][0][1].set(dw)  # fill the gradient

        neon_gdm.optimize([DummyLayer(param_list)], epoch=0)
        (param, grad), states = param_list[0]
        be_W = param.get()[:, 0]
        be_Ws.append(be_W)

        np.testing.assert_allclose(be_W, ng_W, rtol=1e-4)

    print('')