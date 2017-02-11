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
Test of the activation functions
'''
from math import tanh as true_tanh

import pytest
import numpy as np
import ngraph as ng
from ngraph.frontends.neon.activation import (Identity, Rectlin, Rectlinclip,
                                              Softmax, Tanh, Logistic)
from ngraph.testing import ExecutorFactory


class ActivationPair(object):
    tolerance = 0.0

    def reference_value(self, x):
        raise NotImplementedError("Must specify reference activation function")

    def reference_derivative(self, x):
        raise NotImplementedError("Must specify reference activation function")

    def baseline_value(self, x):
        '''
        Use defined ngraph constructed computation to evaluate
        activation on inputs x
        '''
        X = ng.placeholder([ng.make_axis(), ng.make_axis(batch=True)])
        X.axes.set_shape(x.shape)
        with ExecutorFactory() as ex:
            if ex.transformer.transformer_name == "hetr":
                pytest.xfail("unexplained hetr tolerance errors, need to investigate")
            activation_function = ex.executor(self.neon_activation(X), X)
            return activation_function(x)

    def baseline_derivative(self, x):
        X = ng.placeholder([ng.make_axis(), ng.make_axis(batch=True)])
        X.axes.set_shape(x.shape)
        with ExecutorFactory() as ex:
            if ex.transformer.transformer_name == "hetr":
                pytest.xfail("unexplained hetr tolerance errors, need to investigate")
            activation_derivative = ex.derivative(self.neon_activation(X), X)

            # hack to get derivatives
            result = activation_derivative(x)
            result = result.ravel()[0:result.size:(x.size + 1)]
            result = result.reshape(x.shape)

            return result


class IdentityPair(ActivationPair):
    neon_activation = Identity()

    def reference_value(self, x):
        return x

    def reference_derivative(self, x):
        return 1


class RectlinPair(ActivationPair):
    neon_activation = Rectlin()

    def reference_value(self, x):
        return np.maximum(x, 0)

    def reference_derivative(self, x):
        return np.greater(x, 0).astype(np.float32)


class LeakyRectlinPair(ActivationPair):
    slope = 0.2
    neon_activation = Rectlin(slope=0.2)

    def reference_value(self, x):
        return np.maximum(x, 0) + np.minimum(x, 0) * self.slope

    def reference_derivative(self, x):
        return np.greater(x, 0) + np.less(x, 0) * self.slope


class RectlinClipPair(ActivationPair):
    cutoff = 0.2
    neon_activation = Rectlinclip(cutoff=0.2)

    def reference_value(self, x):
        return np.minimum(np.maximum(x, 0), self.cutoff)

    def reference_derivative(self, x):
        return ((x > 0) * (x < self.cutoff)).astype(np.float32)


class TanhPair(ActivationPair):
    neon_activation = Tanh()
    tolerance = 1e-7

    def reference_value(self, x):
        return np.vectorize(true_tanh)(x)

    def reference_derivative(self, x):
        f = self.reference_value(x)
        return (1 - np.square(f))


class LogisticPair(ActivationPair):
    neon_activation = Logistic()
    tolerance = 1e-7

    def reference_value(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def reference_derivative(self, x):
        f = self.reference_value(x)
        return f * (1.0 - f)


class SoftmaxPair(ActivationPair):
    neon_activation = Softmax()
    tolerance = 1e-6

    def reference_value(self, x):
        return (np.exp(x - 1) / np.sum(np.exp(x - 1), axis=0, keepdims=True))

    def reference_derivative(self, x):
        f = self.reference_value(x)
        return f * (1.0 - f)


@pytest.fixture(scope='module',
                params=[
                    IdentityPair(),
                    RectlinPair(),
                    LeakyRectlinPair(),
                    RectlinClipPair(),
                    TanhPair(),
                    LogisticPair(),
                    SoftmaxPair()
                ],
                ids=[
                    'Identity',
                    'Rectlin',
                    'LeakyRectlin',
                    'RectlinClip',
                    'Tanh',
                    'Logistic',
                    'Softmax'
                ])
def activation_pair(request):
    return request.param


@pytest.fixture(scope='module',
                params=[
                    np.array([[1], [3], [2]]),
                    np.array([[0], [1], [-2]]),
                    np.array([[4, 0], [-2, 5]]),
                    np.array([[-1, -3], [-2, -4]])
                ],
                ids=[
                    'all_positive_1d',
                    'mixed_1d',
                    'mixed_2d',
                    'all_negative_2d'
                ])
def all_inputs(request):
    return request.param.astype(np.float32)


def test_activation(all_inputs, activation_pair, transformer_factory):
    ng.testing.assert_allclose(activation_pair.baseline_value(all_inputs),
                               activation_pair.reference_value(all_inputs),
                               rtol=activation_pair.tolerance)


def test_derivative(all_inputs, activation_pair, transformer_factory):
    if (all_inputs.shape[1] != 1 and isinstance(activation_pair, TanhPair)):
        pytest.xfail('Expected tolerance issues for tanh on large-ish values')

    ng.testing.assert_allclose(activation_pair.baseline_derivative(all_inputs),
                               activation_pair.reference_derivative(all_inputs),
                               rtol=activation_pair.tolerance)
