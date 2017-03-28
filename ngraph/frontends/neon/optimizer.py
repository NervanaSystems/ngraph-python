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
import numbers
import ngraph.frontends.common.learning_rate_policies as lrp


def get_learning_rate_policy_callback(lr_params):
    if isinstance(lr_params, numbers.Real):
                # If argument is real number, set policy to fixed and use given value as base_lr
        lr_params = {'name': 'fixed', 'base_lr': lr_params}

    # Check if lr_params contains all required parameters for selected policy.
    if lr_params['name'] not in lrp.lr_policies:
        raise NotImplementedError("Learning rate policy {lr_name} not supported."
                                  "\nSupported policies are: {policies}".format(
                                      lr_name=lr_params['name'],
                                      policies=lrp.lr_policies.keys())
                                  )
    elif all([x in lr_params.keys() for x in lrp.lr_policies[lr_params['name']]['args']]):
        return lrp.lr_policies[lr_params['name']]['obj'](lr_params)
    else:
        raise ValueError("Too few arguments provided to create policy {lr_name}."
                         "\nGiven: {lr_params}"
                         "\nExpected: {lr_args}".format(
                             lr_name=lr_params['name'],
                             lr_params=lr_params.keys(),
                             lr_args=lrp.lr_policies[lr_params['name']])
                         )


def clip_gradient_norm(grad_list, bsz, clip_norm=None):
    """
    Returns a scaling factor to apply to the gradients.

    The scaling factor is computed such that the root mean squared
    average of the scaled gradients across all layers will be less than
    or equal to the provided clip_norm value. This factor is always <1, so
    never scales up the gradients.

    Arguments:
        param_list (list): List of layer parameters
        clip_norm (float, optional): Target norm for the gradients. If not provided
                                     the returned scale_factor will equal 1.
        bsz: the batch size


    Returns:
        Computed scale factor (float)
    """
    if clip_norm is None:
        return 1
    else:
        s = None
        for param in grad_list:
            term = ng.squared_L2(param, out_axes=None)
            if s is None:
                s = term
            else:
                s = s + term
        s = s / bsz
        return clip_norm / ng.maximum(s, clip_norm)


def clip_gradient_value(grad, clip_value=None):
    """
    Element-wise clip a gradient tensor to between ``-clip_value`` and ``+clip_value``.

    Arguments:
        grad (Tensor): List of gradients for a single layer
        clip_value (float, optional): Value to element-wise clip
                                      gradients.
                                      Defaults to None.

    Returns:
        grad (list): List of clipped gradients.
    """
    if clip_value is None:
        return grad
    else:
        return ng.minimum(ng.maximum(grad, -abs(clip_value)), abs(clip_value))


class Optimizer(object):
    """TODO."""
    metadata = {'layer_type': 'optimizer'}

    def __init__(self, name=None, **kwargs):
        super(Optimizer, self).__init__(**kwargs)
        self.name = name

    def update_learning_rate(self):
        pass


class LearningRateOptimizer(Optimizer):

    def __init__(self, learning_rate, iteration=None, **kwargs):
        super(LearningRateOptimizer, self).__init__(**kwargs)
        self.lrate = get_learning_rate_policy_callback(learning_rate)(iteration)


class GradientDescentMomentum(LearningRateOptimizer):
    """
    Stochastic gradient descent with momentum.

    Given the parameters :math:`\\theta`, the learning rate :math:`\\alpha`,
    and the gradients :math:`\\nabla J(\\theta; x)`
    computed on the minibatch data :math:`x`, SGD updates the parameters via

    .. math::
        \\theta' = \\theta - \\alpha\\nabla J(\\theta; x)

    Here we implement SGD with momentum. Momentum tracks the history of
    gradient updates to help the system move faster through saddle points.
    Given the additional parameters: momentum :math:`\gamma`, weight decay :math:`\lambda`,
    and current velocity :math:`v`, we use the following update equations

    .. math::
        v' = \\gamma v - \\alpha(\\nabla J(\\theta; x) + \\lambda\\theta)
        theta' = \\theta + v'

    The optional `nesterov` parameter implements Nesterov Accelerated Gradient.
    If this is set, we use the following update equations instead
    .. math::
        v' = \\gamma^2 v + \\alpha (\\gamma + 1) (\\nabla J(\\theta; x) + \\lambda\\theta)
        theta' = \\theta + v'

    Example usage:

    .. code-block:: python

        import ngraph as ng
        from ngraph.frontends.neon.optimizers import GradientDescentMomentum

        # use SGD with learning rate 0.01 and momentum 0.9, while
        # clipping the gradient magnitude to between -5 and 5.
        loss = ng.squared_l2(actual - estimate)
        opt = GradientDescentMomentum(0.01, 0.9, gradient_clip_value=5)
        updates = opt(loss)
    """
    metadata = {'layer_type': 'gradient_descent_optimizer'}

    def __init__(
            self,
            learning_rate,
            momentum_coef=0.0,
            wdecay=0.0,
            gradient_clip_norm=None,
            gradient_clip_value=None,
            nesterov=False,
            **kwargs):
        super(GradientDescentMomentum, self).__init__(learning_rate=learning_rate, **kwargs)
        self.momentum_coef = momentum_coef
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.wdecay = wdecay
        self.nesterov = nesterov

    @ng.with_op_metadata
    def __call__(self, cost_func):
        all_updates = []
        batch_cost = ng.sum(cost_func, out_axes=())
        batch_size = cost_func.axes.batch_axis().length
        grads = [ng.deriv(batch_cost, v) / batch_size for v in batch_cost.variables()]
        scale_factor = clip_gradient_norm(grads, batch_size, self.gradient_clip_norm)
        for variable, grad in zip(batch_cost.variables(), grads):
            updates = []
            velocity = ng.persistent_tensor(axes=variable.axes,
                                            initial_value=0.).named(variable.name + '_vel')
            clip_grad = clip_gradient_value(grad, self.gradient_clip_value)
            lr = - self.lrate * (scale_factor * clip_grad + self.wdecay * variable)
            updates.append(ng.assign(velocity, velocity * self.momentum_coef + lr))
            if self.nesterov:
                delta = (self.momentum_coef * velocity + lr)
            else:
                delta = velocity
            updates.append(ng.assign(variable, variable + delta))
            all_updates.append(ng.sequential(updates))
        updates = ng.doall(all_updates)
        grads = ng.doall(grads)
        return ng.sequential([grads, updates, 0])


class RMSProp(LearningRateOptimizer):
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
        **kwargs
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

        Notes:
            Only constant learning rate is supported currently.
        """
        super(RMSProp, self).__init__(learning_rate=learning_rate, **kwargs)
        self.state_list = None
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value

    @ng.with_op_metadata
    def __call__(self, cost_func):
        all_updates = []
        batch_cost = ng.sum(cost_func, out_axes=())
        batch_size = cost_func.axes.batch_axis().length

        grads = [ng.deriv(batch_cost, v) / batch_size for v in batch_cost.variables()]
        scale_factor = clip_gradient_norm(grads, batch_size, self.gradient_clip_norm)

        epsilon, decay = (self.epsilon, self.decay_rate)
        for i, (variable, grad) in enumerate(zip(batch_cost.variables(), grads)):
            grad = clip_gradient_value(grad, self.gradient_clip_value)
            state = ng.persistent_tensor(axes=variable.axes, initial_value=0.)
            all_updates.append(ng.sequential([
                ng.assign(state, decay * state + (1.0 - decay) * ng.square(grad)),
                ng.assign(variable, variable - ((scale_factor * grad * self.lrate)
                                                / (ng.sqrt(state + epsilon) + epsilon)))
            ]))

        updates = ng.doall(all_updates)
        grads = ng.doall(grads)
        return ng.sequential([grads, updates, 0])
