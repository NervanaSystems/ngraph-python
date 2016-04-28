
import geon.backends.graph.funs as be
import geon.backends.graph.graph as graph
import numpy as np

@be.with_name_context
def linear(params, x, axes, init):
    params.weights = be.Parameter(axes=x.axes+axes, init=init)
    params.bias = be.Parameter(axes=axes, init=init)
    return be.dot(params.weights, x)+params.bias


def affine(x, activation, axes, init):
    return activation(linear(x, axes, init))


@be.with_name_context
def mlp(params, x, activation, axes_list, init=None):
    value = x
    with be.layers_named('L') as layers:
        for layer, axes in zip(layers, axes_list):
            value = affine(value, activation=activation, axes=axes, init=init)
    return value

def L2(x):
    return be.dot(x,x)


class MyTest(be.Model):
    def __init__(self, width=32, height=32, colors=3, nclasses=10, **kargs):
        super(MyTest, self).__init__(**kargs)

        a = self.a
        self.x = be.input()

        # Hidden layers -- want 2d
        hidden = be.axes_list((a.H.prime(), a.W.prime()), [(14,14)]*2)


        self.value = mlp(self.x, activation=be.tanh, axes_list=hidden + [(a.Y,)])

        self.y = be.input()
        self.error = L2(self.y-self.value)


    def run(self):
        with be.bound_environment(graph=self) as environment:
            a = self.a
            a.Y[10]

            x = np.arange(32*32*3).reshape(32,32,3)

            environment[self.x] = graph.ArrayWithAxes(x, (a.H, a.W, a.C) )
            axes = environment.get_node_axes(self.value)
            print(axes)
            print(self.naming.mlp.L[0].linear.weights)



y = MyTest()
y.run()

print(y)
