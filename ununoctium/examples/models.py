import geon.backends.graph.funs as be
import geon.backends.graph.graph as graph
import numpy as np


@be.with_name_context
def linear(params, x, x_axes, axes, init):
    params.weights = be.Parameter(axes=x_axes+axes, init=init)
    params.bias = be.Parameter(axes=axes, init=init)
    return be.dot(params.weights, x)+params.bias


def affine(x, activation, **kargs):
    return activation(linear(x, **kargs))


@be.with_name_context
def mlp(params, x, activation, x_axes, shape_spec, axes, init=None):
    value = x
    last_axes = x_axes
    with be.layers_named('L') as layers:
        for hidden_axes, hidden_shapes in shape_spec:
            for layer, shape in zip(layers, hidden_shapes):
                layer.axes = tuple(axis.like() for axis in hidden_axes)
                for axis, len in zip(layer.axes, shape):
                    axis[len]
                value = affine(value, activation=activation, x_axes=last_axes, axes=layer.axes, init=init)
                last_axes = layer.axes
        layers.next()
        value = affine(value, activation=activation, x_axes=last_axes, axes=axes, init=init)
    return value

def L2(x):
    return be.dot(x,x)


class MyTest(be.Model):
    def __init__(self, **kargs):
        super(MyTest, self).__init__(**kargs)

        a = self.a
        self.x = be.input()
        self.y = be.input()

        layers=[((a.H, a.W), [(14,14)]*2+[(10,10)])]

        self.value = mlp(self.x, activation=be.tanh, x_axes=self.x.axes, shape_spec=layers, axes=self.y.axes)

        self.error = L2(self.y-self.value)


    def run(self):
        with be.bound_environment(graph=self) as environment:
            a = self.a

            x = np.arange(32*32*3).reshape(32,32,3)
            y = np.arange(10)

            environment[self.x] = graph.ArrayWithAxes(x, (a.H, a.W, a.C) )
            environment[self.y] = graph.ArrayWithAxes(y, (a.Y,))
            axes = environment.get_node_axes(self.value)
            print(axes)
            print(self.naming.mlp.L[0].linear.weights)
            for layer in self.naming.mlp.L:
                try:
                    print(layer.axes)
                except:
                    pass



y = MyTest()
y.run()

print(y)
