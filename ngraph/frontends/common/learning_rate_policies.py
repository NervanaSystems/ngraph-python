# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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

import ngraph as ng
import abc
import collections

# supported lr policies with required parameters
# fixed   = base_lr
# step    = base_lr * gamma ^ (floor(iter / step))
# schedule = base_lr * gamma_1 * gamma_2 * (... up through the number of steps in schedule
# exp     = base_lr * gamma ^ iter
# inv     = base_lr * (1 + gamma * iter) ^ (-power)
# poly    = base_lr * (1 - iter/max_iter) ^ power
# sigmoid = base_lr * (1 / (1 + exp(-gamma * (iter - step_size))))


class lr_policy:

    def __init__(self, name, base_lr):
        self.name = name
        self.base_lr = ng.constant(axes=(), const=base_lr)

    @abc.abstractmethod
    def compute_lr(self, iter):
        pass


class lr_policy_fixed(lr_policy):
    req_args = ('name', 'base_lr',)

    def __init__(self, params):
        lr_policy.__init__(self, params['name'], params['base_lr'])

    def compute_lr(self, iter):
        return self.base_lr


class lr_policy_step(lr_policy):
    req_args = ('name', 'base_lr', 'gamma', 'step')

    def __init__(self, params):
        lr_policy.__init__(self, params['name'], params['base_lr'])
        self.gamma = ng.constant(axes=(), const=params['gamma'])
        self.step = ng.constant(axes=(), const=params['step']) # TODO dtype int?

    def compute_lr(self, iter):
        return self.base_lr * self.gamma ** (iter // self.step)


class lr_policy_schedule(lr_policy):
    req_args = ('name', 'base_lr', 'gamma', 'schedule')

    def __init__(self, params):
        lr_policy.__init__(self, params['name'], params['base_lr'])

        if not isinstance(params['schedule'], collections.Sequence):
            raise ValueError('schedule parameter to schedule policy '
                             'must be a sequence of steps.\n'
                             'Got: {}'.format(params['schedule']))

        num_steps = len(params['schedule'])

        # If gamma is provided as a single value, then just replicate it to
        # match the number of steps in the schedule: e.g. 0.1 --> [0.1, 0.1, 0.1]
        if not isinstance(params['gamma'], collections.Sequence):
            gamma_list = [params['gamma'] for _ in range(num_steps)]
            params['gamma'] = gamma_list

        if len(params['schedule']) != len(params['gamma']):
            raise ValueError('gamma and schedule parameters must have '
                             'same length.  Got {} vs {}'.format(
                                 len(params['gamma']),
                                 len(params['schedule'])
                             ))

        sched_axis = ng.make_axis(length=num_steps, name='schedule')
        self.gamma = ng.constant(axes=[sched_axis], const=params['gamma'])
        self.schedule = ng.constant(axes=[sched_axis], const=params['schedule'], dtype=uint_dtype)

    def compute_lr(self, iteration):
        masked_gamma = (iteration >= self.schedule) * self.gamma
        masked_holes = (iteration < self.schedule)
        return self.base_lr * ng.prod(masked_gamma + masked_holes, out_axes=())


class lr_policy_exp(lr_policy):
    req_args = ('name', 'base_lr', 'gamma')

    def __init__(self, params):
        lr_policy.__init__(self, params['name'], params['base_lr'])
        self.gamma = params['gamma']

    def compute_lr(self, iter):
        return self.base_lr * self.gamma ** iter


class lr_policy_inv(lr_policy):
    req_args = ('name', 'base_lr', 'gamma', 'power')

    def __init__(self, params):
        lr_policy.__init__(self, params['name'], params['base_lr'])
        self.gamma = params['gamma']
        self.power = params['power']

    def compute_lr(self, iter):
        return self.base_lr * (1 + self.gamma * iter) ** (-self.power)


class lr_policy_poly(lr_policy):
    req_args = ('name', 'base_lr', 'max_iter', 'power')

    def __init__(self, params):
        lr_policy.__init__(self, params['name'], params['base_lr'])
        self.max_iter = params['max_iter']
        self.power = params['power']

    def compute_lr(self, iter):
        return self.base_lr * (1 - iter/self.max_iter) ** self.power


class lr_policy_sigmoid(lr_policy):
    req_args = ('name', 'base_lr', 'gamma', 'step_size')

    def __init__(self, params):
        lr_policy.__init__(self, params['name'], params['base_lr'])
        self.gamma = params['gamma']
        self.step_size = params['step_size']

    def compute_lr(self, iter):
        return self.base_lr * (1 / (1 + ng.exp(-self.gamma * (iter - self.step_size))))


lr_policies = {
    'fixed': {'args': lr_policy_fixed.req_args, 'obj': lr_policy_fixed},
    'step': {'args': lr_policy_step.req_args, 'obj': lr_policy_step},
    'schedule': {'args': lr_policy_schedule.req_args, 'obj': lr_policy_schedule},
    'exp': {'args': lr_policy_exp.req_args, 'obj': lr_policy_exp},
    'inv': {'args': lr_policy_inv.req_args, 'obj': lr_policy_inv},
    'poly': {'args': lr_policy_poly.req_args, 'obj': lr_policy_poly},
    'sigmoid': {'args': lr_policy_sigmoid.req_args, 'obj': lr_policy_sigmoid},
}
