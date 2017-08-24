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
from ngraph.frontends.neon import GradientDescentMomentum, RMSProp, Adam, LearningRateOptimizer
from ngraph.testing.execution import ExecutorFactory

pytestmark = pytest.mark.transformer_dependent


atol = rtol = 1e-5


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


class RMSPropReference(object):
    '''
    Simple numpy reference for RMSprop
    '''
    def __init__(self, decay_rate, learning_rate, epsilon):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.state = None

    def __call__(self, input_data, weights):
        '''
        input_data in this case is a numpy array with batch_size on axis 1
        and weights is a matrix with 1 column
        '''
        if self.state is None:
            self.state = np.zeros_like(weights)

        gradient = - input_data.mean(axis=1)

        self.state[:] = self.decay_rate * self.state + \
            (1.0 - self.decay_rate) * np.square(gradient)

        weights[:] = weights \
            - gradient * self.learning_rate / (np.sqrt(self.state + self.epsilon)
                                               + self.epsilon)

        return weights


class AdamReference(object):
    '''
    Simple numpy reference for computing variations of gradient descent for a
    loss = sum(target - weight x input) function
    '''
    def __init__(self, learning_rate, beta_1, beta_2, epsilon):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def __call__(self, input_data, weights):
        '''
        input_data in this case is a numpy array with batch_size on axis 1
        and weights is a matrix with 1 column
        '''
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        self.t += 1
        gradient = -input_data.mean(axis=1)
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * gradient * gradient
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)
        weights = weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return weights


def compare_optimizer(opt_ng, opt_ref):

    # Set up data placeholders
    C = ng.make_axis(20)
    N = ng.make_axis(32, name='N')

    data = ng.placeholder([C, N])
    target = ng.placeholder([N])

    # params to be updated using optimizer to be tested
    np_W = np.random.rand(C.length)
    W = ng.variable([C], initial_value=np_W)

    # Set up op graph
    cost = ng.sum(target - ng.dot(W, data), out_axis=())
    updated_weights = ng.sequential([opt_ng(cost), W])

    # Set up the computation and run the "train" loop
    with ExecutorFactory() as ex:
        opt_ng_comp = ex.transformer.computation(updated_weights, data, target)
        mock_dataset = data_generator(20, C.length, N.length)

        for x, y in mock_dataset:
            ng_W = opt_ng_comp(x, y)  # updated weights for ngraph optimizer
            np_W = opt_ref(x, np_W)   # updated weights for reference optimizer

            ng.testing.assert_allclose(np_W, ng_W, rtol=1e-3)


def compare_optimizer_variable_select(opt_ng, opt_ref):

    # Set up data placeholders
    C = ng.make_axis(20)
    N = ng.make_axis(32, name='N')

    data = ng.placeholder([C, N])
    target = ng.placeholder([N])

    # params to be updated using optimizer to be tested
    np_W1 = np.random.rand(C.length)
    np_W2 = np.random.rand(C.length)
    W1 = ng.variable([C], initial_value=np_W1)
    W2 = ng.variable([C], initial_value=np_W2)

    # Set up op graph
    cost = ng.sum(target - ng.dot(W1, data) - ng.dot(W2, data), out_axis=())
    updated_weights = ng.sequential([opt_ng(cost, variables=[W1]), W1])

    # Set up the computation and run the "train" loop
    with ExecutorFactory() as ex:
        opt_ng_comp = ex.transformer.computation([updated_weights, W2], data, target)
        mock_dataset = data_generator(20, C.length, N.length)

        for x, y in mock_dataset:
            [ng_W1, ng_W2] = opt_ng_comp(x, y)  # updated weights for ngraph optimizer
            np_W1 = opt_ref(x, np_W1)   # updated weights for reference optimizer

            ng.testing.assert_allclose(np_W1, ng_W1, rtol=1e-3)
            ng.testing.assert_allclose(np_W2, ng_W2, rtol=1e-3)


def data_generator(iteration_count, C, N):
    for i in range(iteration_count):
        yield (np.random.rand(C, N).astype('float32'),
               np.random.rand(N).astype('float32'))


# generate fixtures this way so that collection names are deterministic and can
# be run in parallel:
# https://github.com/pytest-dev/pytest/issues/594
@pytest.fixture(params=[0, 1])
def random_learning_rate():
    return np.random.random()


@pytest.fixture(params=[0, 1, 2, 3])
def random_momentum_coef():
    return np.random.random()


@pytest.config.flex_disabled(reason="The most cases fail because of too strict assert tolerance")
@pytest.mark.parametrize("wdecay", [0.0005, 0.000, 0.001, 0.1])
@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("select_variables", [False, True])
def test_gdm(random_learning_rate, random_momentum_coef, wdecay, nesterov,
             select_variables):

    # Setup the baseline and reference optimizers to be tested
    gdm_args = {'learning_rate': random_learning_rate,
                'momentum_coef': random_momentum_coef,
                'wdecay': wdecay,
                'nesterov': nesterov}

    gdm_ref = GDMReference(**gdm_args)
    gdm = GradientDescentMomentum(**gdm_args)

    # test baseline against reference
    if select_variables:
        compare_optimizer_variable_select(gdm, gdm_ref)
    else:
        compare_optimizer(gdm, gdm_ref)


@pytest.config.flex_disabled(reason="Results totally mismatch")
@pytest.mark.parametrize("decay_rate", [0.95, 1])
@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("select_variables", [False, True])
def test_rmsprop(random_learning_rate, decay_rate, epsilon, select_variables):
    rmsprop_args = {'learning_rate': random_learning_rate,
                    'epsilon': epsilon,
                    'decay_rate': decay_rate}

    rmsprop_ref = RMSPropReference(**rmsprop_args)
    rms = RMSProp(**rmsprop_args)

    # test baseline against reference
    if select_variables:
        compare_optimizer_variable_select(rms, rmsprop_ref)
    else:
        compare_optimizer(rms, rmsprop_ref)


@pytest.fixture(params=[0, 1])
def random_beta_1():
    return np.random.uniform(low=0.0, high=1.0)


@pytest.fixture(params=[0, 1])
def random_beta_2():
    return np.random.uniform(low=0.0, high=1.0)


@pytest.config.argon_disabled  # TODO triage
@pytest.config.flex_skip(reason="Usually all cases fail but very rarely some pass for flex - "
                                "because of the random character of the parameters")
@pytest.mark.parametrize("epsilon", [1e-8])
@pytest.mark.parametrize("select_variables", [False, True])
def test_adam(random_learning_rate, random_beta_1, random_beta_2, epsilon, select_variables):

    # Setup the baseline and reference optimizers to be tested
    adam_args = {'learning_rate': random_learning_rate,
                 'beta_1': random_beta_1,
                 'beta_2': random_beta_2,
                 'epsilon': epsilon}

    adam_reference = AdamReference(**adam_args)
    adam = Adam(**adam_args)

    # test baseline against reference
    if select_variables:
        compare_optimizer_variable_select(adam, adam_reference)
    else:
        compare_optimizer(adam, adam_reference)


@pytest.config.argon_disabled  # TODO triage
@pytest.config.flex_disabled(reason="Unknown problem yet")
def test_learning_policy_step():
    base_learning_rate = 1.0
    drop_factor = 0.1
    step = 20

    lr_params = {'name': 'step',
                 'base_lr': base_learning_rate,
                 'gamma': drop_factor,
                 'step': step}

    iteration = ng.placeholder((), dtype=np.dtype(np.uint32))
    lro = LearningRateOptimizer(learning_rate=lr_params, iteration=iteration)

    with ExecutorFactory() as ex:
        stepped_learning_rate = ex.transformer.computation(lro.lrate, iteration)

        for iter_input in [10, 50, 90, 6, 15]:
            baseline_value = stepped_learning_rate(iter_input)
            reference_value = base_learning_rate * (drop_factor ** (iter_input // step))

            ng.testing.assert_allclose(baseline_value, reference_value, rtol=1e-5)


def test_learning_policy_fixed_with_input():
    base_learning_rate = 0.1

    iteration = ng.placeholder((), dtype=np.dtype(np.uint32))
    lro = LearningRateOptimizer(learning_rate=base_learning_rate, iteration=iteration)

    with ExecutorFactory() as ex:
        fixed_learning_rate = ex.transformer.computation(lro.lrate, iteration)

        for iter_input in [10, 50, 90, 6, 15]:
            baseline_value = fixed_learning_rate(iter_input)

            ng.testing.assert_allclose(baseline_value, base_learning_rate, rtol=1e-6)


def test_learning_policy_fixed_without_input():
    base_learning_rate = 0.1

    lro = LearningRateOptimizer(learning_rate=base_learning_rate)

    with ExecutorFactory() as ex:
        fixed_learning_rate = ex.transformer.computation(lro.lrate)
        baseline_value = fixed_learning_rate()
        ng.testing.assert_allclose(baseline_value, base_learning_rate, rtol=1e-6)


@pytest.config.argon_disabled  # TODO triage
@pytest.mark.parametrize("drop_factor", [0.1,
                                         [0.1, 0.2, 0.3, 0.4, 0.5]])
def test_learning_policy_schedule(drop_factor):
    base_learning_rate = 1.0
    schedule = [20, 100, 300, 750, 1000]

    lr_params = {'name': 'schedule',
                 'base_lr': base_learning_rate,
                 'gamma': drop_factor,
                 'schedule': schedule}

    iteration = ng.placeholder((), dtype=np.dtype(np.uint32))
    lro = LearningRateOptimizer(learning_rate=lr_params, iteration=iteration)

    schedule.append(np.inf)
    np_schedule = np.array(schedule)

    with ExecutorFactory() as ex:
        scheduled_learning_rate = ex.transformer.computation(lro.lrate, iteration)

        for iter_input in np.random.randint(0, 1100, 5):
            baseline_value = scheduled_learning_rate(iter_input)
            max_step_ind = np.where(iter_input < np_schedule)[0][0]
            if isinstance(drop_factor, list):
                scale_factor = np.prod(drop_factor[:max_step_ind])
            else:
                scale_factor = drop_factor ** max_step_ind
            reference_value = base_learning_rate * scale_factor
            ng.testing.assert_allclose(baseline_value, reference_value, rtol=1e-5)


if __name__ == '__main__':
    test_rmsprop(0.1, 0.95, 1e-6)
    test_gdm(0.1, 0.1, 0.1, False)
    test_adam(0.1, 0.5, 0.9, 1e-6, None)
