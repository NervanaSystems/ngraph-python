import geon.backends.graph.funs as be
from neon.optimizers.optimizer import Schedule, StepSchedule, PowerSchedule, ExpSchedule, PolySchedule


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


class GradientDescent(Optimizer):
    def __init__(self, learning_rate, gradient_clip_norm=None, gradient_clip_value=None, schedule=ExpSchedule(1.0), **kargs):
        super(GradientDescent, self).__init__(**kargs)
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value

    def configure(self, cost):
        self.learning_rate_placeholder = be.placeholder(axes=())
        learning_rate_value = self.learning_rate_placeholder
        params = cost.parameters()
        grads = [be.deriv(cost, param) for param in params]
        if self.gradient_clip_norm:
            learning_rate_value = learning_rate_value * clip_gradient_norm(grads)
        if self.gradient_clip_value is not None:
            grads = [clip_gradient_value(param, self.gradient_clip_value) for grade in grads]
        self.updates = be.doall(all=[be.assign(param, param - learning_rate_value * grad) for param, grad in zip(params, grads)])
        return self.updates

    def optimize(self, epoch):
        learning_rate = self.schedule.get_learning_rate(self.learning_rate, epoch)
        self.learning_rate_placeholder.value = learning_rate



