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
from __future__ import division
from builtins import object, zip
import ngraph as ng
import numpy as np


class Schedule(object):
    """
    Learning rate schedule.

    By default implements a constant learning rate:

    .. code-block:: python

        # Constant learning rate of 0.01 across training iterations
        optimizer = GradientDescentMomentum(0.01, 0.9, schedule = Schedule())

    Otherwise, the schedule multiplies the learning rate by change at every element in
    ``step_config``.
    For example,

    .. code-block:: python

        schedule = Schedule(step_config=[2, 6], change=0.5)
        optimizer = GradientDescentMomentum(1.0, 0.9, schedule = Schedule())

    will yield a learning rate schedule of:

    .. csv-table::
        :header: "iteration", "LR"
        :widths: 20, 10

        0, 1.0
        1, 1.0
        2, 0.5
        3, 0.5
        4, 0.5
        5, 0.5
        6, 0.25
        7, 0.25
        8, 0.25
        9, 0.25
    """
    def __init__(self, step_config=None, change=1.):
        """
        Class constructor.

        Arguments:
            step_config (list, optional): Configure the step times (list of epoch indices).
                                          Defaults to None (constant).
            change (int, optional): The learning rate is
                                    multiplied by ``change ** steps``, where ``steps`` is the
                                    number of steps in the step schedule that have passed.
        """

        if isinstance(step_config, list) and isinstance(change, list):
            assert len(step_config) == len(change), "change and step_config must have the same" \
                "length after step_config is deduplicated to do epoch-level LR assignment."

        self.step_config = step_config
        self.change = change
        self.steps = 0

    def get_learning_rate(self, learning_rate, epoch):
        """
        Returns the current learning rate given the epoch and initial learning rate.

        Arguments:
            learning_rate (float): Initial learning rate
            epoch (int): Current epoch, used to calculate the adjusted learning rate

        Returns:
            (float): The adjusted learning rate
        """
        if isinstance(self.step_config, list):
            self.steps = np.sum(epoch >= np.array(self.step_config))

        return learning_rate * self.change ** self.steps


def clip_gradient_norm(grad_list, bsz, clip_norm=None):
    """
    TODO.

    Arguments:
      grad_list: TODO
      clip_norm: TODO
      bsz: TODO

    Returns:

    """
    if clip_norm:
        s = None
        for param in grad_list:
            term = ng.squared_L2(param)
            if s is None:
                s = term
            else:
                s = s + term
        s = s / bsz
        return clip_norm / ng.max(s, clip_norm)
    else:
        return 1


def clip_gradient_value(grad, clip_value=None):
    """
    TODO.

    Arguments:
      grad: TODO
      clip_value: TODO

    Returns:

    """
    if clip_value:
        return ng.minimum(ng.maximum(grad, -abs(clip_value)), abs(clip_value))
    else:
        return grad


class Optimizer(object):
    """TODO."""
    metadata = {'layer_type': 'optimizer'}

    def __init__(self, name=None, **kwargs):
        super(Optimizer, self).__init__(**kwargs)
        self.name = name
        self.iteration_index = 0


class GradientDescentMomentum(Optimizer):
    """TODO."""
    metadata = {'layer_type': 'gradient_descent_optimizer'}

    def __init__(
            self,
            learning_rate,
            momentum_coef=0.0,
            stochastic_round=False,
            wdecay=0.0,
            gradient_clip_norm=None,
            gradient_clip_value=None,
            name=None,
            schedule=Schedule(),
            **kwargs):
        super(GradientDescentMomentum, self).__init__(**kwargs)
        self.momentum_coef = momentum_coef
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.wdecay = wdecay
        self.schedule = schedule
        self.stochastic_round = stochastic_round
        self.learning_rate = ng.persistent_tensor(axes=(),
                                                  initial_value=learning_rate).named('lrate')

    @ng.with_op_metadata
    def __call__(self, cost_func):
        with ng.Op.saved_user_deps():
            velocity_updates, param_updates = [], []
            batch_cost = ng.sum(cost_func, out_axes=())
            batch_size = cost_func.axes.batch_axes()[0].length
            scale_factor = 1

            for variable in batch_cost.variables():
                grad = clip_gradient_value(ng.deriv(batch_cost, variable) / batch_size,
                                           self.gradient_clip_value)

                velocity = ng.persistent_tensor(axes=variable.axes,
                                                initial_value=0.).named(variable.name + '_vel')
                velocity_updates.append(
                    ng.assign(velocity,
                              velocity * self.momentum_coef - self.learning_rate * (
                                  scale_factor * grad + self.wdecay * variable)))

                param_updates.append(ng.assign(variable, variable + velocity))

            lr_update = [ng.assign(self.learning_rate,
                                   self.schedule.get_learning_rate(self.learning_rate,
                                                                   self.iteration_index))]

            updates = ng.doall(velocity_updates + param_updates + lr_update)
            self.iteration_index += 1

        return updates


class RMSProp(Optimizer):
    """
    Root Mean Square propagation.
    Root Mean Square (RMS) propagation protects against vanishing and
    exploding gradients. In RMSprop, the gradient is divided by a running
    average of recent gradients. Given the parameters :math:`\\theta`, gradient :math:`\\nabla J`,
    we keep a running average :math:`\\mu` of the last :math:`1/\\lambda` gradients squared.
    The update equations are then given by
    .. math::
        \\mu' &= \\lambda\\mu + (1-\\lambda)(\\nabla J)^2
    .. math::
        \\theta' &= \\theta - \\frac{\\alpha}{\\sqrt{\\mu + \\epsilon} + \\epsilon}\\nabla J
    where we use :math:`\\epsilon` as a (small) smoothing factor to prevent from dividing by zero.
    """
    metadata = {'layer_type': 'RMS_prop_optimizer'}

    def __init__(
        self,
        decay_rate=0.95,
        learning_rate=2e-3,
        epsilon=1e-6,
        gradient_clip_norm=None,
        gradient_clip_value=None,
        name=None,
        schedule=Schedule()
    ):
        """
        Class constructor.
        Arguments:
            decay_rate (float): decay rate of states
            learning_rate (float): the multiplication coefficent of updates
            epsilon (float): smoothing epsilon to avoid divide by zeros
            gradient_clip_norm (float, optional): Target gradient norm.
                                                  Defaults to None.
            gradient_clip_value (float, optional): Value to element-wise clip
                                                   gradients.
                                                   Defaults to None.
            schedule (neon.optimizers.optimizer.Schedule, optional): Learning rate schedule.
                                                                     Defaults to a constant.
        Notes:
            Only constant learning rate is supported currently.
        """
        super(RMSProp, self).__init__(name=name)
        self.state_list = None
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.schedule = schedule
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.learning_rate = ng.persistent_tensor(axes=(),
                                                  initial_value=learning_rate).named('lrate')

    @ng.with_op_metadata
    def __call__(self, cost_func):
        with ng.Op.saved_user_deps():
            state_updates, param_updates = [], []
            batch_cost = ng.sum(cost_func, out_axes=())
            batch_size = cost_func.axes.batch_axes()[0].length

            grads = [ng.deriv(batch_cost, v) / batch_size for v in batch_cost.variables()]
            scale_factor = clip_gradient_norm(grads, batch_size, self.gradient_clip_norm)

            epsilon, decay = (self.epsilon, self.decay_rate)
            for i, (variable, grad) in enumerate(zip(batch_cost.variables(), grads)):
                grad = clip_gradient_value(grad, self.gradient_clip_value)

                state = ng.persistent_tensor(axes=variable.axes, initial_value=0.)
                state_updates.append(
                    ng.assign(
                        lvalue=state,
                        rvalue=decay * state + (1.0 - decay) * ng.square(grad)
                    ).named('state_u_%s' % i)
                )

                param_updates.append(
                    ng.assign(
                        lvalue=variable,
                        rvalue=variable - ((scale_factor * grad * self.learning_rate)
                                           / (ng.sqrt(state + epsilon) + epsilon)),
                    ).named('var_u_%s' % i)
                )

            lr_update = [ng.assign(self.learning_rate,
                                   self.schedule.get_learning_rate(self.learning_rate,
                                                                   self.iteration_index))]

            updates = ng.doall(state_updates + param_updates + lr_update)
            self.iteration_index += 1

        return updates
