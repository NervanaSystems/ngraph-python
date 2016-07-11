import geon.backends.graph.funs as be
from neon.optimizers.optimizer import Schedule, StepSchedule, PowerSchedule, ExpSchedule, PolySchedule
from neon.initializers import Constant


# Optimizer support
def L2(x):
    return be.dot(x, x)


def clip_gradient_norm(grad_list, clip_norm, bsz):
    s = None
    for param in grad_list:
        term = be.sqrt(L2(param))
        if s is None:
            s = term
        else:
            s = s + term
    s = s/bsz
    return clip_norm/be.max(s, clip_norm)


def clip_gradient_value(grad, clip_value=None):
    if clip_value:
        return be.clip(grad, -abs(clip_value), abs(clip_value))
    else:
        return grad


class Optimizer(object):
    def __init__(self, name=None, **kargs):
        super(Optimizer, self).__init__(**kargs)
        self.name = name

    def configure(self, cost):
        raise NotImplementedError()

    def optimize(self, params_to_optimize, epoch):
        raise NotImplementedError()


class GradientDescentMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_coef=0.0, stochastic_round=False,
                 wdecay=0.0, gradient_clip_norm=None, gradient_clip_value=None,
                 name=None, schedule=ExpSchedule(1.0), **kargs):
        super(GradientDescentMomentum, self).__init__(**kargs)
        self.learning_rate = learning_rate
        self.momentum_coef = momentum_coef
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.wdecay = wdecay
        self.schedule = schedule
        self.stochastic_round = stochastic_round

    def configure(self, cost):
        self.learning_rate_placeholder = be.placeholder(axes=(), name='lrate') 
        learning_rate_value = self.learning_rate_placeholder
        params = cost.parameters()
        grads = [be.deriv(cost, param) for param in params]
        velocities = [be.Temporary(axes=param.axes, init=Constant(0)) for param in params]

        scale_factor = 1
        if self.gradient_clip_norm:
            scale_factor = clip_gradient_norm(grads)
        if self.gradient_clip_value is not None:
            grads = [clip_gradient_value(param, self.gradient_clip_value) for grade in grads]

        velocity_updates = [
            be.assign(
                lvalue = velocity,
                rvalue =  velocity * self.momentum_coef - learning_rate_value * (scale_factor * grad + self.wdecay * param)
            )
        for param, grad, velocity in zip(params, grads, velocities)]

        param_updates = [be.assign(lvalue=param, rvalue=param+velocity) for param, velocity in zip(params, velocities)]
        return be.doall(velocity_updates + param_updates)

    def optimize(self, epoch):
        learning_rate = self.schedule.get_learning_rate(self.learning_rate, epoch)
        self.learning_rate_placeholder.value = learning_rate
