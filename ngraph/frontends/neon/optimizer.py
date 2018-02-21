# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
from __future__ import division, absolute_import
import logging
import numpy as np
import ngraph as ng
import numbers
import ngraph.frontends.common.learning_rate_policies as lrp
from ngraph.frontends.neon.graph import SubGraph

logger = logging.getLogger(__name__)


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
                                     the returned scale_factor will equal 1

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
        clip_value (float, optional): Value to element-wise clip gradients. Default: no clipping

    Returns:
        grad (list): List of clipped gradients.
    """
    if clip_value is None:
        return grad
    else:
        return ng.minimum(ng.maximum(grad, -abs(clip_value)), abs(clip_value))


def clip_weight_value(weight, clip_value=None, min_value_override=None):
    """
    Element-wise clip a weight tensor to between ``min_value_override`` and ``clip_value``.

    Arguments:
        weight (Tensor): List of gradients for a single layer
        clip_value (float, optional): Value to element-wise clip weights. Default: no clipping
        min_value (float, optional): Value to minimum value to element-wise clip
                                     weights. Default: -abs(clip_value)

    Returns:
        weight (list): List of clipped weights.
    """
    if clip_value is None:
        return weight
    else:
        if min_value_override is None:
            min_value_override = -abs(clip_value)
        return ng.minimum(ng.maximum(weight, min_value_override), abs(clip_value))


class Optimizer(SubGraph):
    """TODO."""

    def __init__(self, name=None, **kwargs):
        super(Optimizer, self).__init__(name=name, **kwargs)

    def update_learning_rate(self):
        pass


class LearningRateOptimizer(Optimizer):
    """
    Base class for a gradient-based optimizer

    Arguments:
        learning_rate (float): Multiplicative coefficient to scale gradients before the updates
                               are applied
        iteration (placeholder, optional): Placeholder op used to store the current training
                                           iteration for the purposes of using learning rate
                                           schedulers. Default: None
        gradient_clip_norm (float, optional): Target norm for the gradients. Default: no clipping
        gradient_clip_value (float, optional): Value to element-wise clip gradients.
                                               Default: no clipping
        weight_clip_value (float, optional): Value to element-wise clip weights after updates are
                                             applied, symmetric around 0. Default: no clipping
    """

    def __init__(self, learning_rate, iteration=0,
                 gradient_clip_norm=None,
                 gradient_clip_value=None,
                 weight_clip_value=None,
                 **kwargs):
        super(LearningRateOptimizer, self).__init__(**kwargs)
        self.lrate = get_learning_rate_policy_callback(learning_rate)(iteration)
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.weight_clip_value = weight_clip_value

    @SubGraph.scope_op_creation
    def __call__(self, cost_func, variables=None, subgraph=None, warning=False):
        """
        Arguments:
            cost_func (Op): The cost function to optimize
            variables (list of variables): List of variables to optimize
            subgraph (SubGraph): A subgraph instance containing all variables to optimize
            warning (bool): If True displays warning message if any variables
                            specified do not participate in batch cost computation

        .. Note::
            If subgraph is provided, the variables to optimize will be taken from it.
            Otherwise, they can be provided explicitly by passing a list as `variables`.
            If neither `subgraph` nor `variables` is provided, the variables to optimize will be
            all trainable variables on which `cost` depends.
        """

        all_updates = []
        batch_cost = ng.sum(cost_func, out_axes=())
        if cost_func.axes.batch_axis() is None:
            batch_size = 1
        else:
            batch_size = cost_func.axes.batch_axis().length

        # determine variables to optimize
        if subgraph is not None:
            if variables is not None:
                raise ValueError("variables and subgraph cannot both be specified.")
            variables = list(subgraph.variables.values())

        if variables is None:
            variables = batch_cost.variables()
        elif variables is not None and warning is True:
            all_variables = batch_cost.variables()
            selected_variables = all_variables & set(variables)
            if len(selected_variables) < len(variables):
                logger.warn("not all selected variables participate in cost computation")

        # gradients
        grads = [ng.deriv(batch_cost, v) / batch_size for v in variables]
        scale_factor = clip_gradient_norm(grads, self.gradient_clip_norm)

        # updates
        for variable, grad in zip(variables, grads):
            updates = self.variable_update(variable, grad, scale_factor)
            all_updates.append(updates)
        updates = ng.doall(all_updates)
        grads = ng.doall(grads)
        clips = ng.doall([ng.assign(variable, clip_weight_value(variable, self.weight_clip_value))
                          for variable in variables])
        return ng.sequential([grads, updates, clips, 0])


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

    Arguments:
        learning_rate (float): Multiplicative coefficient to scale gradients before the updates
                               are applied
        momentum_coef (float, optional): Coefficient of momentum. Default: 0
        wdecay (float, optional): Amount of weight decay. Default: 0
        nesterov (bool, optional): Use nesterov accelerated gradient. Default: False
        iteration (placeholder, optional): Placeholder op used to store the current training
                                           iteration for the purposes of using learning rate
                                           schedulers. Default: None
        gradient_clip_norm (float, optional): Target norm for the gradients. Default: no clipping
        gradient_clip_value (float, optional): Value to element-wise clip gradients.
                                               Default: no clipping
        weight_clip_value (float, optional): Value to element-wise clip weights after updates are
                                             applied, symmetric around 0. Default: no clipping

    Examples:
    .. code-block:: python

        import ngraph as ng
        from ngraph.frontends.neon.optimizers import GradientDescentMomentum

        # use SGD with learning rate 0.01 and momentum 0.9, while
        # clipping the gradient magnitude to between -5 and 5.
        loss = ng.squared_l2(actual - estimate)
        opt = GradientDescentMomentum(0.01, 0.9, gradient_clip_value=5)
        updates = opt(loss)
    """

    def __init__(
            self,
            learning_rate,
            momentum_coef=0.0,
            wdecay=0.0,
            gradient_clip_norm=None,
            gradient_clip_value=None,
            weight_clip_value=None,
            nesterov=False,
            **kwargs):
        super(GradientDescentMomentum, self).__init__(learning_rate=learning_rate,
                                                      gradient_clip_norm=gradient_clip_norm,
                                                      gradient_clip_value=gradient_clip_value,
                                                      weight_clip_value=weight_clip_value,
                                                      **kwargs)
        self.momentum_coef = momentum_coef
        self.wdecay = wdecay
        self.nesterov = nesterov

    def variable_update(self, variable, grad, scale_factor):
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
        # Old way without Kahan summation
        # updates.append(ng.assign(variable, variable + delta))

        # New way with Kahan summation
        kahan_c = ng.persistent_tensor(axes=variable.axes,
                                       initial_value=0.).named(variable.name + '_c')
        kahan_y = ng.persistent_tensor(axes=variable.axes,
                                       initial_value=0.).named(variable.name + '_y')
        kahan_t = ng.persistent_tensor(axes=variable.axes,
                                       initial_value=0.).named(variable.name + '_t')
        updates.append(ng.assign(kahan_y, delta - kahan_c))
        updates.append(ng.assign(kahan_t, variable + kahan_y))
        updates.append(ng.assign(kahan_c, (kahan_t - variable) - kahan_y))
        updates.append(ng.assign(variable, kahan_t))
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

    Arguments:
        decay_rate (float): decay rate of states. Default: 0.95
        learning_rate (float): the multiplication coefficent of updates
        epsilon (float): smoothing epsilon to avoid divide by zeros. Default: 1e-6
        iteration (placeholder, optional): Placeholder op used to store the current training
                                           iteration for the purposes of using learning rate
                                           schedulers. Default: None
        gradient_clip_norm (float, optional): Target norm for the gradients. Default: no clipping
        gradient_clip_value (float, optional): Value to element-wise clip gradients.
                                               Default: no clipping
        weight_clip_value (float, optional): Value to element-wise clip weights after updates are
                                             applied, symmetric around 0. Default: no clipping
        wdecay (float, optional): Amount of weight decay (L2) penalty. Default: 0
        momentum_coef (float, optional): Coefficient of momentum. Default: 0
    """

    def __init__(
        self,
        decay_rate=0.95,
        learning_rate=2e-3,
        epsilon=1e-6,
        gradient_clip_norm=None,
        gradient_clip_value=None,
        weight_clip_value=None,
        wdecay=0.0,
        momentum_coef=0.0,
        **kwargs
    ):
        super(RMSProp, self).__init__(learning_rate=learning_rate,
                                      gradient_clip_norm=gradient_clip_norm,
                                      gradient_clip_value=gradient_clip_value,
                                      weight_clip_value=weight_clip_value,
                                      **kwargs)
        self.state_list = None
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.wdecay = wdecay
        self.momentum = momentum_coef

    def variable_update(self, variable, grad, scale_factor):
        epsilon, decay = (self.epsilon, self.decay_rate)
        grad = clip_gradient_value(grad, self.gradient_clip_value)
        state = ng.persistent_tensor(axes=variable.axes, initial_value=1.)
        velocity = ng.persistent_tensor(axes=variable.axes,
                                        initial_value=0.).named(variable.name + '_vel')
        updates = ng.sequential([
            ng.assign(state, decay * state + (1.0 - decay) * ng.square(grad)),
            ng.assign(velocity, velocity * self.momentum +
                      (self.lrate * scale_factor * grad / ng.sqrt(state + epsilon)) +
                      self.lrate * self.wdecay * variable),
            ng.assign(variable, variable - velocity)
        ])
        return updates


class Adam(LearningRateOptimizer):
    """
    Adam optimizer

    The Adam optimizer combines features from RMSprop and Adagrad. We
    accumulate both the first and second moments of the gradient with decay
    rates :math:`\\beta_1` and :math:`\\beta_2` corresponding to window sizes of
    :math:`1/\\beta_1` and :math:`1/\\beta_2`, respectively.

    .. math::
        m' &= \\beta_1 m + (1-\\beta_1) \\nabla J

    .. math::
        v' &= \\beta_2 v + (1-\\beta_2) (\\nabla J)^2

    We update the parameters by the ratio of the two moments:

    .. math::
        \\theta = \\theta - \\alpha \\frac{\\hat{m}'}{\\sqrt{\\hat{v}'}+\\epsilon}

    where we compute the bias-corrected moments :math:`\\hat{m}'` and :math:`\\hat{v}'` via

    .. math::
        \\hat{m}' &= m'/(1-\\beta_1^t)

    .. math::
        \\hat{v}' &= v'/(1-\\beta_1^t)

    Arguments:
        learning_rate (float): the multiplication coefficient of updates
        beta_1 (float): decay of 1st order moment. Default: .9
        beta_2 (float): decay of 2nd order moment. Default: .999
        epsilon (float): numerical stability factor. Default: 1e-8
        iteration (placeholder, optional): Placeholder op used to store the current training
                                           iteration for the purposes of using learning rate
                                           schedulers. Default: None
        gradient_clip_norm (float, optional): Target norm for the gradients. Default: no clipping
        gradient_clip_value (float, optional): Value to element-wise clip gradients.
                                               Default: no clipping
        weight_clip_value (float, optional): Value to element-wise clip weights after updates are
                                             applied, symmetric around 0. Default: no clipping
    """

    def __init__(
            self,
            learning_rate=1e-3,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            gradient_clip_norm=None,
            gradient_clip_value=None,
            weight_clip_value=None,
            **kwargs
    ):
        super(Adam, self).__init__(learning_rate,
                                   gradient_clip_norm=gradient_clip_norm,
                                   gradient_clip_value=gradient_clip_value,
                                   weight_clip_value=weight_clip_value,
                                   **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    @SubGraph.scope_op_creation
    def __call__(self, *args, **kwargs):
        if len(self.ops) == 0:
            self.beta_1 = ng.constant(self.beta_1, dtype=np.float32)
            self.beta_2 = ng.constant(self.beta_2, dtype=np.float32)
            self.t = ng.persistent_tensor(axes=(), initial_value=0)

        self.t = ng.sequential([ng.assign(self.t, self.t + 1), self.t])
        self.ell = self.lrate * ng.sqrt(1 - self.beta_2 ** self.t) / (1 - self.beta_1 ** self.t)

        return super(Adam, self).__call__(*args, **kwargs)

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


class Adagrad(LearningRateOptimizer):
    """
    Adagrad optimization algorithm.

    Adagrad is an algorithm that adapts the learning rate individually for each parameter
    by dividing by the :math:`L_2`-norm of all previous gradients. Given the parameters
    :math:`\\theta`, gradient :math:`\\nabla J`, accumulating norm :math:`G`, and smoothing
    factor :math:`\\epsilon`, we use the update equations:

    .. math::

        G' = G + (\\nabla J)^2

    .. math::

        \\theta' = \\theta - \\frac{\\alpha}{\sqrt{G' + \\epsilon}} \\nabla J

    where the smoothing factor :math:`\\epsilon` prevents from dividing by zero.
    By adjusting the learning rate individually for each parameter, Adagrad adapts
    to the geometry of the error surface. Differently scaled weights have appropriately scaled
    update steps.

    Arguments:
        learning_rate (float): Multiplicative coefficient to scale gradients before the updates
                               are applied
        epsilon (float, optional): Numerical stability factor. Default: 1e-8
        iteration (placeholder, optional): Placeholder op used to store the current training
                                           iteration for the purposes of using learning rate
                                           schedulers. Default: None
        gradient_clip_norm (float, optional): Target norm for the gradients. Default: no clipping
        gradient_clip_value (float, optional): Value to element-wise clip gradients.
                                               Default: no clipping
        weight_clip_value (float, optional): Value to element-wise clip weights after updates are
                                             applied, symmetric around 0. Default: no clipping
    Examples:
    .. code-block:: python

        import ngraph as ng
        from ngraph.frontends.neon.optimizers import Adagrad

        # use Adagrad with a learning rate of 1e-3
        optimizer = Adagrad(learning_rate=1e-3, epsilon=1e-8)
    """

    def __init__(
            self,
            learning_rate=1e-3,
            epsilon=1e-8,
            gradient_clip_norm=None,
            gradient_clip_value=None,
            weight_clip_value=None,
            **kwargs
    ):
        super(Adagrad, self).__init__(learning_rate,
                                      gradient_clip_norm=gradient_clip_norm,
                                      gradient_clip_value=gradient_clip_value,
                                      weight_clip_value=weight_clip_value,
                                      **kwargs)
        self.epsilon = epsilon

    def variable_update(self, variable, grad, scale_factor):
        grad = clip_gradient_value(grad, self.gradient_clip_value)
        state = ng.persistent_tensor(axes=grad.axes, initial_value=0.)
        updates = ng.sequential([
            ng.assign(state, state + ng.square(grad)),
            ng.assign(variable,
                      variable - (scale_factor * self.lrate * grad)
                      / (ng.sqrt(state + self.epsilon)))
        ])
        return updates
