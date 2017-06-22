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
from ngraph.frontends.neon import Layer, Linear, Affine, Convolution, \
    BiRNN, LSTM, ConstantInit, Rectlin, Tanh
from ngraph.frontends.neon.optimizer import RMSProp
from ngraph.testing import ExecutorFactory, RandomTensorGenerator


rng = RandomTensorGenerator(0, np.float32)


class LayerClass(object):

    def __init__(self):
        super(LayerClass, self).__init__()

    def get_layer(self):
        return self.layer

    def get_weights(self):
        return (self.layer.W, )

    def get_input(self):
        N = ng.make_axis(name='N', length=4)
        return ng.placeholder(axes=ng.make_axes(N))


class LinearLayer(LayerClass):

    def __init__(self):
        super(LinearLayer, self).__init__()
        self.layer = Linear(ConstantInit(0.0), nout=10)


class AffineLayer(LayerClass):

    def __init__(self):
        super(AffineLayer, self).__init__()
        self.layer = Affine(ConstantInit(0.0), nout=10,
                            bias_init=ConstantInit(0.0), activation=Rectlin())

    def get_weights(self):
        return (self.layer.linear.W, self.layer.bias.W)


class ConvolutionLayer(LayerClass):

    def __init__(self):
        super(ConvolutionLayer, self).__init__()
        self.layer = Convolution({'T': 1, 'R': 1, 'S': 1, 'K': 1},
                                 ConstantInit(0.0),
                                 {'str_d': 1, 'str_h': 1, 'str_w': 1},
                                 {'pad_d': 0, 'pad_h': 0, 'pad_w': 0},
                                 {'dil_d': 1, 'dil_h': 1, 'dil_w': 1},
                                 bias_init=ConstantInit(0.0),
                                 batch_norm=True)

    def get_weights(self):
        return (self.layer.conv.W, self.layer.bias.W,
                self.layer.batch_norm.gamma, self.layer.batch_norm.beta)

    def get_input(self):
        ax_i = ng.make_axes([
            ng.make_axis(name='C', length=1),
            ng.make_axis(name='H', length=8),
            ng.make_axis(name='W', length=8),
            ng.make_axis(name='D', length=1),
            ng.make_axis(name='N', length=4)
        ])
        return ng.placeholder(axes=ax_i)


class LSTMLayer(LayerClass):

    def __init__(self):
        super(LSTMLayer, self).__init__()
        self.layer = LSTM(nout=16, init=ConstantInit(0.0), activation=Tanh(),
                          gate_activation=Tanh())

    def get_weights(self):
        return tuple(list(self.layer.W_input.values()) + list(self.layer.W_recur.values()) +
                     list(self.layer.b.values()))

    def get_input(self):
        ax_i = ng.make_axes([ng.make_axis(name='M', length=4),
                             ng.make_axis(name='REC', length=4),
                             ng.make_axis(name='N', length=4)])
        return ng.placeholder(axes=ax_i)


class BiRNNLayer(LayerClass):

    def __init__(self):
        super(BiRNNLayer, self).__init__()
        self.layer = BiRNN(nout=16, init=ConstantInit(0.0), activation=Tanh(), concat_out=True)

    def get_weights(self):
        def rnn_weights(rnn):
            return rnn.W_input, rnn.W_recur, rnn.b
        return rnn_weights(self.layer.fwd_rnn) + rnn_weights(self.layer.bwd_rnn)

    def get_input(self):
        ax_i = ng.make_axes([ng.make_axis(name='M', length=4),
                             ng.make_axis(name='REC', length=4),
                             ng.make_axis(name='N', length=4)])
        return ng.placeholder(axes=ax_i)


@pytest.fixture(scope='module',
                params=[
                    LinearLayer,
                    AffineLayer,
                    ConvolutionLayer,
                    LSTMLayer,
                    BiRNNLayer,
                ],
                ids=[
                    'Linear',
                    'Affine',
                    'Convolution',
                    'LSTM',
                    'BiRNN',
                ])
def layer_cls(request):
    return request.param


# test for new "scope" implementation - Layers have associated subgraphs
@pytest.fixture(scope='module',
                params=['subgraph',
                        'variables'])
def optimizer_scope_keyword(request):
    return request.param


def test_selective_optimization(transformer_factory, layer_cls,
                                optimizer_scope_keyword):

    def make_network():
        # create layer
        layer = layer_cls()

        # fprop
        x = layer.get_input()
        y = layer.get_layer()(x)
        t = ng.placeholder(axes=y.axes)
        cost = ng.squared_L2(y - t)
        return layer, cost, x, t

    def make_optimizer():
        return RMSProp()

    with ExecutorFactory() as ex:

        # optimization specifying subgraph or variables
        layerA, costA, xA, tA = make_network()
        optimizerA = make_optimizer()
        if optimizer_scope_keyword == 'subgraph':
            updatesA = optimizerA(costA, subgraph=[layerA.get_layer()])
        elif optimizer_scope_keyword == 'variables':
            updatesA = optimizerA(costA, variables=list(layerA.get_layer().variables))
        compA = ex.executor(updatesA, xA, tA)

        # optimization not specifying subgraph or variables
        layerB, costB, xB, tB = make_network()
        optimizerB = make_optimizer()
        updatesB = optimizerB(costB)
        compB = ex.executor(updatesB, xB, tB)

        x_val = rng.uniform(-1.0, 1.0, xA.axes)
        t_val = rng.uniform(-1.0, 1.0, tA.axes)

        compA(x_val, t_val)
        compB(x_val, t_val)

        for wA, wB in zip(layerA.get_weights(), layerB.get_weights()):
            wA_val = ex.get_tensor_view_value(wA)
            wB_val = ex.get_tensor_view_value(wB)
            #assert (wA_val == wB_val).all()
            ng.testing.assert_allclose(wA_val, wB_val)
