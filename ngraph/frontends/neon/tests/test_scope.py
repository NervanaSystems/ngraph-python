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
import pytest
import numpy as np
import ngraph as ng
from ngraph.frontends.neon.layer import Affine
from ngraph.frontends.neon.model import Sequential
from ngraph.frontends.neon.activation import Rectlin, Logistic
from ngraph.frontends.neon.initializer import ConstantInit
from ngraph.frontends.neon.optimizer import GradientDescentMomentum
from ngraph.testing import ExecutorFactory, RandomTensorGenerator


rng = RandomTensorGenerator(0, np.float32)


# self.W is None check - test that self.W doesn't get recreated?

# TODO: fixtures for
# layer types
# optimizers

# make this into different classes for each optimizer, for fixture
def make_optimizer():
    return GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.0, stochastic_round=False,
                                        wdecay=0.0, gradient_clip_norm=None,
                                        gradient_clip_value=None)


def test_scope_2layer(transformer_factory):
    # input
    F = ng.make_axis(length=8)
    N = ng.make_axis(length=4).named('N')
    axes = ng.make_axes([F, N])

    # make up inputs and targets
    inputs = rng.uniform(-1.0, 1.0, axes)
    targets = np.array([0, 1, 1, 0])

    # initial weight values 
    nout1 = 16
    Wlin1 = np.random.normal(0, 1, (nout1, F.length))
    Wbias1 = 0.0
    Wlin2 = np.random.normal(0, 1, (1, nout1))
    Wbias2 = 0.0
    W_lin_init = [Wlin1, Wlin2]
    W_bias_init = [Wbias1, Wbias2]

    def make_network(scope1=None, scope2=None):
        # input
        x = ng.placeholder(axes)
        # targets
        t = ng.placeholder(ng.make_axes([N]))
        # 2 layer network, each layer has its own scope
        layer1 = Affine(ConstantInit(val=Wlin1), nout=nout1, bias_init=ConstantInit(val=Wbias1),
                        activation=Rectlin(), batch_norm=False, scope=scope2)
        layer2 = Affine(ConstantInit(val=Wlin2), nout=1, bias_init=ConstantInit(val=Wbias2),
                        activation=Logistic(), batch_norm=False, scope=scope2)
        seq = Sequential([layer1, layer2])
        # loss
        p_t = seq(x)
        loss = ng.cross_entropy_binary(p_t, t)
        return seq, x, t, loss

    # get all updates, without scope, for ground truth
    seq_all, x_all, t_all, loss_all = make_network()
    optimizer_all = make_optimizer()
    updates_all = optimizer_all(loss_all)

    # updates with scope, for same network
    seq_scope, x_scope, t_scope, loss_scope = make_network(scope1='s1', scope2='s2')
    optimizer_s1 = make_optimizer()
    optimizer_s2 = make_optimizer()
    updates_s1 = optimizer_s1(loss_scope, variable_scope='s1')
    updates_s2 = optimizer_s2(loss_scope, variable_scope='s2')

    update_scope, other_scope = 's1', 's2'
    with ExecutoryFactory() as ex:

        # variables that should not be updated in scoped network
        update_layer = int(update_scope[-1]) - 1
        no_update_layer = int(other_scope[-1]) - 1

        # ground truth
        ex.executor(updates_all, x_all, t_all)(inputs, targets)

        # update only variables in scope
        ex.executor(updates_s1, x_scope, t_scope)(inputs, targets)

        # check variables not in scope have not changed
        assert seq_scope.layers[no_update_layer].linear.W == W_lin_init[no_update_layer]
        assert seq_scope.layers[no_update_layer].bias.W == W_bias_init[no_update_layer]
        # check variables in scope have correct updated values
        assert seq_scope.layers[update_layer].linear.W == seq_all.layers[update_layer].linear.W
        assert seq_scope.layers[update_layer].bias.W == seq_all.layers[update_layer].bias.W

        print ('passed')



test_scope_2layer(None)
