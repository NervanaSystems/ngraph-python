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
import ngraph as ng
from ngraph.frontends.neon import Layer, Linear, Affine, Convolution, \
    BiRNN, LSTM, ConstantInit, Rectlin, Tanh


class LayerClass(object):

    def __init__(self):
        super(LayerClass, self).__init__()

    def get_layer(self):
        return self.layer

    def get_weights(self):
        return (self.layer.W, )

    def get_input(self):
        return ng.placeholder(axes=())


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
                                 {'dil_d': 0, 'dil_h': 0, 'dil_w': 0},
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
            ng.make_axis(name='N', length=4)
        ])
        return ng.placeholder(axes=ax_i)


class LSTMLayer(LayerClass):

    def __init__(self):
        super(LSTMLayer, self).__init__()
        self.layer = LSTM(nout=16, init=ConstantInit(0.0), activation=Tanh(),
                          gate_activation=Tanh())

    def get_weights(self):
        return tuple(self.layer.W_input.values() + self.layer.W_recur.values() +
                     self.layer.b.values())

    def get_input(self):
        ax_i = ng.make_axes([ng.make_axis(name='M', length=8),
                             ng.make_axis(name='REC', length=10),
                             ng.make_axis(name='N', length=4)])
        return ng.placeholder(axes=ax_i)


class BiRNNLayer(LayerClass):

    def __init__(self):
        super(BiRNNLayer, self).__init__()
        self.layer = BiRNN(nout=16, init=ConstantInit(0.0), activation=Tanh())

    def get_weights(self):
        def rnn_weights(rnn):
            return rnn.W_input, rnn.W_recur, rnn.b
        return rnn_weights(self.layer.fwd_rnn) + rnn_weights(self.layer.bwd_rnn)

    def get_input(self):
        ax_i = ng.make_axes([ng.make_axis(name='M', length=8),
                             ng.make_axis(name='REC', length=10),
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


def test_scope(layer_cls):

    scope1 = 's1'
    layer0 = layer_cls()
    with Layer.variable_scope(scope1):
        assert Layer.active_scope == scope1
        layer1 = layer_cls()
    layer2 = layer_cls()

    # have to call layers to initialize W
    x = layer0.get_input()
    layer0.get_layer()(x)
    layer1.get_layer()(x)
    layer2.get_layer()(x)

    for w in layer0.get_weights():
        assert w.scope is None
    for w in layer1.get_weights():
        assert w.scope == scope1, "found scope {} instead of {}".format(w.scope, scope1)
    for w in layer2.get_weights():
        assert w.scope is None
    print "{} tests passed".format(layer_cls)
