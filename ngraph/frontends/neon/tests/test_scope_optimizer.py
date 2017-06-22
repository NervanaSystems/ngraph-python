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
from ngraph.frontends.neon.optimizer import GradientDescentMomentum, RMSProp, Adam
from ngraph.testing import ExecutorFactory, RandomTensorGenerator


rng = RandomTensorGenerator(0, np.float32)


def gradient_descent_momentum():
    return GradientDescentMomentum(0.1)


def rmsprop():
    return RMSProp()


def adam():
    return Adam()


@pytest.fixture(scope='module',
                params=[gradient_descent_momentum,
                        rmsprop,
                        adam])
def optimizer_factory(request):
    return request.param


@pytest.fixture(scope='module',
                params=['subgraph',
                        'variables'])
def optimizer_scope_keyword(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0, 1])
def update_layer(request):
    return request.param


def test_scope_2layer(optimizer_factory, optimizer_scope_keyword, update_layer,
                      transformer_factory):
    """
    Two layer network with each layer in a different variable scope.
    Test that optimizing with respect to variables in one scope correctly
    updates the variables in that scope and leaves the other layer variables
    unchanged.
    """

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

    def make_network():
        # 2 layer network, each layer has its own scope
        x = ng.placeholder(axes)  # inputs
        t = ng.placeholder(ng.make_axes([ng.make_axis(length=1), N]))  # targets
        layer1 = Affine(ConstantInit(val=Wlin1), nout=nout1,
                        bias_init=ConstantInit(val=Wbias1),
                        activation=Rectlin(), batch_norm=False)
        layer2 = Affine(ConstantInit(val=Wlin2), nout=1,
                        bias_init=ConstantInit(val=Wbias2),
                        activation=Logistic(), batch_norm=False)
        seq = Sequential([layer1, layer2])
        p_t = seq(x)
        t_cast = ng.cast_axes(t, p_t.axes)  # TODO: how can this be avoided?
        loss = ng.cross_entropy_binary(p_t, t_cast)
        return seq, x, t, loss

    with ExecutorFactory() as ex:

        # layer indices
        no_update_layer = 1 - update_layer

        # get all updates, without scope, for ground truth
        seq_all, x_all, t_all, loss_all = make_network()
        optimizer_all = optimizer_factory()
        updates_all = optimizer_all(loss_all)
        result_list = [updates_all]
        result_list.append(seq_all.layers[update_layer].linear.W)
        result_list.append(seq_all.layers[update_layer].bias.W)
        network_all = ex.executor(result_list, x_all, t_all)

        # update only variables in one scope, for same network
        seq_scope, x_scope, t_scope, loss_scope = make_network()
        optimizer_scope = optimizer_factory()
        updates_scope = optimizer_scope(loss_scope, variable_scope=update_scope)
        result_list = [updates_scope]
        result_list.append(seq_scope.layers[update_layer].linear.W)
        result_list.append(seq_scope.layers[update_layer].bias.W)
        result_list.append(seq_scope.layers[no_update_layer].linear.W)
        result_list.append(seq_scope.layers[no_update_layer].bias.W)
        network_scope = ex.executor(result_list, x_scope, t_scope)

        # one iteration of weight updates
        (_, Wtruth_linear_all, Wtruth_bias_all) = network_all(inputs, targets)
        (_, Wactual_linear_scope_update,
         Wactual_bias_scope_update,
         Wactual_linear_scope_no_update,
         Wactual_bias_scope_no_update) = network_scope(inputs, targets)

        # def get_np_ary(seq, layer_ind, layer_type):
        #     return ex.get_tensor_view_value(getattr(seq.layers[layer_ind], layer_type).W)

        # check variables not in scope have not changed
        assert np.all(Wactual_linear_scope_no_update == W_lin_init[no_update_layer])
        assert np.all(Wactual_bias_scope_no_update == W_bias_init[no_update_layer])

        # check variables in scope have correct updated values
        assert ng.testing.allclose(Wactual_linear_scope_update, Wtruth_linear_all)
        assert ng.testing.allclose(Wactual_bias_scope_update, Wtruth_bias_all)
