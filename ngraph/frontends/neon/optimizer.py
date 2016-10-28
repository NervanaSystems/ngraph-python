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
