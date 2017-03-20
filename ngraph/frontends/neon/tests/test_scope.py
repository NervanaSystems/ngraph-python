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
from ngraph.frontends.neon import Layer, Linear, Affine, Convolution, LSTM, \
    ConstantInit, Rectlin, Tanh


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


# TODO: test different Layer types
@pytest.fixture(scope='module',
                params=[
                    LinearLayer,
                    AffineLayer,
                ],
                ids=[
                    'Linear',
                    'Affine',
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
    # TODO: is this how we envision use? scope is defined at creation of Layer, not
    # scope of subsequent call
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
