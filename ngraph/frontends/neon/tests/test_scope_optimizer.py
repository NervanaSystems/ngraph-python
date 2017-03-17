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
from ngraph.frontends.neon.layer import Layer, Affine
from ngraph.frontends.neon.model import Sequential
from ngraph.frontends.neon.activation import Rectlin, Logistic
from ngraph.frontends.neon.initializer import ConstantInit
from ngraph.frontends.neon.optimizer import GradientDescentMomentum
from ngraph.testing import ExecutorFactory, RandomTensorGenerator


rng = RandomTensorGenerator(0, np.float32)

# TODO: test different layer types
# TODO: test different optimizer classes, once they support scope
# self.W is None check - test that self.W doesn't get recreated?
def make_optimizer():
    return GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.0, stochastic_round=False,
                                        wdecay=0.0, gradient_clip_norm=None,
                                        gradient_clip_value=None)


@pytest.fixture(scope='module',
                params=[
                    ('s1', 's2'),
                    ('s2', 's1'),
                ])
def scope_pair(request):
    return request.param


def test_scope_2layer(scope_pair, transformer_factory):

    # this test peeks at values of layer weights, not hetr-compatible
    if transformer_factory.name == 'hetr':
        pytest.xfail("Hetr is expected to fail with code that checks side-effects")

    # input
    F = ng.make_axis(length=8)
    N = ng.make_axis(length=4).named('N')
    axes = ng.make_axes([F, N])

    # make up inputs and targets
    inputs = rng.uniform(-1.0, 1.0, axes)
    targets = np.array([0, 1, 1, 0])

    # initial weight values 
    nout1 = 16
    Wlin1 = np.random.normal(0, 1, (nout1, F.length)).astype(np.float32)
    Wbias1 = 0.0
    Wlin2 = np.random.normal(0, 1, (1, nout1)).astype(np.float32)
    Wbias2 = 0.0
    W_lin_init = [Wlin1, Wlin2]
    W_bias_init = [Wbias1, Wbias2]

    def make_network(scope1=None, scope2=None):
        # 2 layer network, each layer has its own scope
        x = ng.placeholder(axes)  # inputs
        t = ng.placeholder(ng.make_axes([ng.make_axis(length=1), N]))  # targets
        with Layer.variable_scope(scope1):
            layer1 = Affine(ConstantInit(val=Wlin1), nout=nout1,
                            bias_init=ConstantInit(val=Wbias1),
                            activation=Rectlin(), batch_norm=False)
        with Layer.variable_scope(scope2):
            layer2 = Affine(ConstantInit(val=Wlin2), nout=1,
                            bias_init=ConstantInit(val=Wbias2),
                            activation=Logistic(), batch_norm=False)
        seq = Sequential([layer1, layer2])
        p_t = seq(x)
        t_cast = ng.cast_axes(t, p_t.axes)  # TODO: how can this be avoided?
        loss = ng.cross_entropy_binary(p_t, t_cast)
        return seq, x, t, loss

    update_scope, no_update_scope = scope_pair
    with ExecutorFactory() as ex:

        # layer indices
        update_layer = int(update_scope[-1]) - 1
        no_update_layer = int(no_update_scope[-1]) - 1

        # get all updates, without scope, for ground truth
        seq_all, x_all, t_all, loss_all = make_network()
        optimizer_all = make_optimizer()
        updates_all = optimizer_all(loss_all)
        network_all = ex.executor(updates_all, x_all, t_all)

        # update only variables in one scope, for same network
        seq_scope, x_scope, t_scope, loss_scope = make_network(scope1='s1', scope2='s2')
        optimizer_scope = make_optimizer()
        updates_scope = optimizer_scope(loss_scope, variable_scope=update_scope)
        network_scope = ex.executor(updates_scope, x_scope, t_scope)

        # one iteration of weight updates
        network_all(inputs, targets)
        network_scope(inputs, targets)

        def get_np_ary(seq, layer_ind, layer_type):
            return getattr(seq.layers[layer_ind], layer_type).W.value.get(None)

        # check variables not in scope have not changed
        Wactual = get_np_ary(seq_scope, no_update_layer, 'linear')
        assert np.all(Wactual == W_lin_init[no_update_layer])
        Wactual = get_np_ary(seq_scope, no_update_layer, 'bias')
        assert np.all(Wactual == W_bias_init[no_update_layer])

        # check variables in scope have correct updated values
        Wactual = get_np_ary(seq_scope, update_layer, 'linear')
        Wtruth = get_np_ary(seq_all, update_layer, 'linear')
        assert ng.testing.allclose(Wactual, Wtruth)
        Wactual = get_np_ary(seq_scope, update_layer, 'bias')
        Wtruth = get_np_ary(seq_all, update_layer, 'bias')
        assert ng.testing.allclose(Wactual, Wtruth)
