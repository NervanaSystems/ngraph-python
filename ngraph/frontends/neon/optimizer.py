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
import numpy as np
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


def clip_gradient_norm(grad_list, clip_norm=None):
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

        s = ng.sqrt(s)
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

    @ng.with_op_metadata
    def __call__(self, cost_func, variable_scope=None):
        self._pre_call_hook()
        all_updates = []
        batch_cost = ng.sum(cost_func, out_axes=())
        if cost_func.axes.batch_axis() is None:
            batch_size = 1
        else:
            batch_size = cost_func.axes.batch_axis().length

        selected_variables = batch_cost.variables()
        if variable_scope is not None:
            selected_variables = [op for op in selected_variables if op.scope == variable_scope]
        grads = [ng.deriv(batch_cost, v) / batch_size for v in selected_variables]
        scale_factor = clip_gradient_norm(grads, self.gradient_clip_norm)

        for variable, grad in zip(selected_variables, grads):
            updates = self.variable_update(variable, grad, scale_factor)
            all_updates.append(updates)
        updates = ng.doall(all_updates)
        grads = ng.doall(grads)
        return ng.sequential([grads, updates, 0])

    def _pre_call_hook(self):
        pass


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

    def variable_update(self, variable, grad, scale_factor):
        updates = []
        velocity = ng.persistent_tensor(axes=variable.axes,
                                        initial_value=0.).named(variable.name + '_vel')
        # add metadata to the gradient node indicating that
        # it should be reduced across data-parallel workers before used for optimization
        grad.metadata['reduce_func'] = 'sum'
        clip_grad = clip_gradient_value(grad, self.gradient_clip_value)
        lr = - self.lrate * (scale_factor * clip_grad + self.wdecay * variable)
        updates.append(ng.assign(velocity, velocity * self.momentum_coef + lr))
        if self.nesterov:
            delta = (self.momentum_coef * velocity + lr)
        else:
            delta = velocity
        updates.append(ng.assign(variable, variable + delta))
        return ng.sequential(updates)


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

    def variable_update(self, variable, grad, scale_factor):
        epsilon, decay = (self.epsilon, self.decay_rate)
        grad = clip_gradient_value(grad, self.gradient_clip_value)
        state = ng.persistent_tensor(axes=variable.axes, initial_value=0.)
        updates = ng.sequential([
            ng.assign(state, decay * state + (1.0 - decay) * ng.square(grad)),
            ng.assign(variable, variable - ((scale_factor * grad * self.lrate)
                                            / (ng.sqrt(state + epsilon) + epsilon)))
        ])
        return updates


class Adam(LearningRateOptimizer):
    """
    Adam optimizer

    TODO docstring

    """
    metadata = {'layer_type': 'adam_optimizer'}

    def __init__(
        self,
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        gradient_clip_norm=None,
        gradient_clip_value=None,
        **kwargs
    ):
        """
        Class constructor.
        Arguments:
            learning_rate (float): the multiplication coefficient of updates
            beta_1 (float): decay of 1st order moment
            beta_2 (float): decay of 2nd order moment
            epsilon (float): numerical stability factor
            gradient_clip_norm (float, optional): Target gradient norm.
                                                  Defaults to None.
            gradient_clip_value (float, optional): Value to element-wise clip
                                                   gradients.
                                                   Defaults to None.
        """
        super(Adam, self).__init__(learning_rate, **kwargs)
        self.beta_1 = ng.constant(beta_1, dtype=np.float32)
        self.beta_2 = ng.constant(beta_2, dtype=np.float32)
        self.epsilon = epsilon
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.t = ng.persistent_tensor(axes=(), initial_value=0)

    def _pre_call_hook(self):
        self.t = ng.sequential([ng.assign(self.t, self.t + 1), self.t])
        self.ell = self.lrate * ng.sqrt(1 - self.beta_2**self.t) / (1 - self.beta_1**self.t)

    def variable_update(self, variable, grad, scale_factor):
        m = ng.persistent_tensor(axes=grad.axes, initial_value=0.)
        v = ng.persistent_tensor(axes=grad.axes, initial_value=0.)
        updates = ng.sequential([
            ng.assign(m, m * self.beta_1 + (1 - self.beta_1) * grad),
            ng.assign(v, v * self.beta_2 + (1 - self.beta_2) * grad * grad),
            ng.assign(variable,
                      variable - (scale_factor * self.ell * m) / (ng.sqrt(v) + self.epsilon))
        ])
        return updates
