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
from neon.optimizers.optimizer import Schedule
from neon.initializers import Constant


# Optimizer support
def L2(x):
    """
    TODO.

    Arguments:
      x: TODO

    Returns:

    """
    return ng.dot(x, x)


def clip_gradient_norm(grad_list, clip_norm, bsz):
    """
    TODO.

    Arguments:
      grad_list: TODO
      clip_norm: TODO
      bsz: TODO

    Returns:

    """
    s = None
    for param in grad_list:
        term = ng.sqrt(L2(param))
        if s is None:
            s = term
        else:
            s = s + term
    s = s / bsz
    return clip_norm / ng.max(s, clip_norm)


def clip_gradient_value(grad, clip_value=None):
    """
    TODO.

    Arguments:
      grad: TODO
      clip_value: TODO

    Returns:

    """
    if clip_value:
        return ng.clip(grad, -abs(clip_value), abs(clip_value))
    else:
        return grad


class Optimizer(object):
    """TODO."""

    def __init__(self, name=None, **kwargs):
        super(Optimizer, self).__init__(**kwargs)
        self.name = name

    def configure(self, cost):
        """
        Returns the update and computation subgraph `Op` to optimize a cost
        function.

        Arguments:
          cost: The cost `Op` that this optimizer is attempting to minimize.

        Returns:
          An `Op` implementing the parameter updates to `Variable`s upstream of
          `cost`.

        """
        raise NotImplementedError()

    def optimize(self, params_to_optimize, epoch):
        """
        TODO.

        Arguments:
          params_to_optimize: TODO
          epoch: TODO

        Returns:

        """
        raise NotImplementedError()


class GradientDescentMomentum(Optimizer):
    """TODO."""

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
        self.learning_rate = learning_rate
        self.momentum_coef = momentum_coef
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.wdecay = wdecay
        self.schedule = schedule
        self.stochastic_round = stochastic_round

    def configure(self, cost):
        """
        TODO.

        Arguments:
          cost: TODO
          batch_size: TODO

        Returns:

        """
        self.learning_rate_placeholder = ng.placeholder(axes=(), name='lrate')
        learning_rate_value = self.learning_rate_placeholder
        variables = list(cost.variables())
        grads = [
            ng.deriv(cost, variable)
            for variable in variables
        ]
        velocities = [ng.persistent_tensor(
            axes=variable.axes, init=Constant(0),
            name=variable.name + '_vel') for variable in variables]

        scale_factor = 1
        if self.gradient_clip_norm:
            scale_factor = clip_gradient_norm(grads)
        if self.gradient_clip_value is not None:
            grads = [clip_gradient_value(
                grade, self.gradient_clip_value) for grade in grads]

        velocity_updates = [
            ng.assign(
                lvalue=velocity,
                rvalue=velocity * self.momentum_coef - learning_rate_value * (
                    scale_factor * grad + self.wdecay * variable)
            )
            for variable, grad, velocity in zip(variables, grads, velocities)]

        param_updates = [
            ng.assign(lvalue=variable, rvalue=variable + velocity)
            for variable, velocity in zip(variables, velocities)
        ]

        return ng.doall(velocity_updates + param_updates)

    def optimize(self, epoch):
        """
        TODO.

        Arguments:
          epoch: TODO

        Returns:

        """
        learning_rate = self.schedule.get_learning_rate(
            self.learning_rate, epoch)
        self.learning_rate_placeholder.value[()] = learning_rate


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
    def __init__(
        self,
        stochastic_round=False,
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
            stochastic_round (bool): Set this to True for stochastic rounding.
                                     If False rounding will be to nearest.
                                     If True will perform stochastic rounding using default width.
                                     Only affects the gpu backend.
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
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.stochastic_round = stochastic_round
    def configure(self, cost):
        self.lrate = ng.placeholder(axes=(), name='lrate')

        variables = list(cost.variables())
        grads = [ng.deriv(cost, variable) / 50.0 for variable in variables]
        scale_factor = 1
        if self.gradient_clip_norm:
            scale_factor = clip_gradient_norm(grads)
        if self.gradient_clip_value is not None:
            grads = [clip_gradient_value(
                variable, self.gradient_clip_value) for grade in grads]

        epsilon, decay = (self.epsilon, self.decay_rate)
        states = [
            ng.temporary(axes=variable.axes, init=Constant(0))
            for variable in variables
        ]
        state_updates = [
            ng.assign(
                lvalue=state,
                rvalue=decay * state + (1.0 - decay) * ng.square(grad),
                name='state_u_%s' % i
            ) for state, grad, i in zip(states, grads, range(len(states)))
        ]
        param_updates = [
            ng.assign(
                lvalue=param,
                rvalue=param
                - (scale_factor * grad * self.lrate)
                    / (ng.sqrt(state + epsilon) + epsilon),
                name='param_u_%s' % i
            ) for state, grad, param, i in zip(states, grads, variables, range(len(states)))
        ]
        return ng.doall(state_updates + param_updates)

    def optimize(self, epoch):

        """
        TODO.
        Arguments:
          epoch: TODO
        Returns:
        """
        learning_rate = self.schedule.get_learning_rate(
            self.learning_rate, epoch)
        self.lrate.value[()] = learning_rate

