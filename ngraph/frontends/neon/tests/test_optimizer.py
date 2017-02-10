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
import pytest
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import GradientDescentMomentum
from ngraph.testing.execution import ExecutorFactory


class GDMReference(object):
    '''
    Simple numpy reference for computing variations of gradient descent for a
    loss = sum(target - weight x input) function
    '''
    def __init__(self, learning_rate, momentum_coef, wdecay, nesterov):
        self.learning_rate = learning_rate
        self.momentum_coef = momentum_coef
        self.wdecay = wdecay
        self.nesterov = nesterov
        self.velocity = None

    def __call__(self, input_data, weights):
        '''
        input_data in this case is a numpy array with batch_size on axis 1
        and weights is a matrix with 1 column
        '''
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)

        gradient = self.wdecay * weights - input_data.mean(axis=1)
        self.velocity[:] = self.momentum_coef * self.velocity - self.learning_rate * gradient

        if self.nesterov:
            weights[:] = (weights + self.momentum_coef * self.velocity -
                          self.learning_rate * gradient)
        else:
            weights[:] = weights + self.velocity

        return weights


# generate fixtures this way so that collection names are deterministic and can
# be run in parallel:
# https://github.com/pytest-dev/pytest/issues/594
@pytest.fixture(params=[0, 1])
def random_learning_rate():
    return np.random.random()


@pytest.fixture(params=[0, 1, 2, 3])
def random_momentum_coef():
    return np.random.random()


@pytest.mark.parametrize("wdecay", [0.0005, 0.000, 0.001, 0.1])
@pytest.mark.parametrize("nesterov", [False, True])
def test_gdm(random_learning_rate, random_momentum_coef, wdecay, nesterov, transformer_factory):

    # Setup the baseline and reference optimizers to be tested
    gdm_args = {'learning_rate': random_learning_rate,
                'momentum_coef': random_momentum_coef,
                'wdecay': wdecay,
                'nesterov': nesterov}

    gdm_reference = GDMReference(**gdm_args)
    gdm = GradientDescentMomentum(**gdm_args)

    # Set up data placeholders
    C = ng.make_axis(20)
    N = ng.make_axis(32, batch=True)

    data = ng.placeholder([C, N])
    target = ng.placeholder([N])

    # params to be updated using GDM
    np_W = np.random.rand(C.length)
    W = ng.variable([C - 1], initial_value=np_W)

    # Set up op graph
    cost = ng.sum(target - ng.dot(W, data), out_axis=())
    updated_weights = ng.sequential([gdm(cost), W])

    def data_generator(iteration_count):
        for i in range(iteration_count):
            yield (np.random.rand(C.length, N.length).astype('float32'),
                   np.random.rand(N.length).astype('float32'))

    # Set up the computation and run the "train" loop
    with ExecutorFactory() as ex:
        gdm_baseline = ex.transformer.computation(updated_weights, data, target)
        mock_dataset = data_generator(20)

        for x, y in mock_dataset:
            ng_W = gdm_baseline(x, y)  # updated weights for ngraph optimizer
            np_W = gdm_reference(x, np_W)  # updated weights for reference optimizer

            ng.testing.assert_allclose(np_W, ng_W, rtol=1e-3)
