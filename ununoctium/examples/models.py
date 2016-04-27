
import geon.backends.graph.funs as be

import geon.backends.graph.typing as typing
import geon.backends.graph.graph as graph
import numpy as np


def linear(x, axes, init):
    weights = be.Parameter(axes=x.axes+axes, init=init)
    bias = be.Parameter(axes=axes, init=init)
    return be.dot(weights, x)+bias


def affine(x, activation, axes, init):
    return activation(linear(x, axes, init))


def mlp(x, activation, axes_list, init=None):
    value = x
    for axes in axes_list:
        value = affine(value, activation=activation, axes=axes, init=init)
    return value


class MyTest(be.Model):
    def __init__(self, width=32, height=32, colors=3, nclasses=10, **kargs):
        super(MyTest, self).__init__(**kargs)

        a = self.a
        self.x = be.input()

        # Hidden layers -- want 2d
        hidden = be.axes_list((a.H.prime(), a.W.prime()), [(14,14)]*2)


        self.value = mlp(self.x, activation=be.tanh, axes_list=hidden + [(a.Y,)])

    def run(self):
        with be.bound_environment() as environment:
            a = self.a
            a.Y[10]

            x = np.arange(32*32*3).reshape(32,32,3)

            environment[self.x] = graph.ArrayWithAxes(x, (a.H, a.W, a.C) )
            axes = self.value.axes.evaluate(environment)
            print(axes)



y = MyTest()
y.run()

print(y)
