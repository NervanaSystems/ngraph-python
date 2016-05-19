import geon.backends.graph.defmod as be

class Uniform(object):
    def __init__(self, low=0.0, high=1.0):
        self.low = low
        self.high = high

    def __call__(self, evaluator, value):
        evaluator.uniform(value, self.low, self.high)


@be.with_name_context
def linear(params, x, x_axes, axes, batch_axes=(), init=None):
    params.weights = be.Parameter(axes=axes+x_axes-batch_axes, init=init)
    params.bias = be.Parameter(axes=axes, init=init)
    return be.dot(params.weights, x)+params.bias


def affine(x, activation, **kargs):
    return activation(linear(x, **kargs))


@be.with_name_context
def mlp(params, x, activation, x_axes, shape_spec, axes, **kargs):
    value = x
    last_axes = x_axes
    with be.layers_named('L') as layers:
        for hidden_axes, hidden_shapes in shape_spec:
            for layer, shape in zip(layers, hidden_shapes):
                layer.axes = tuple(be.Axis(like=axis) for axis in hidden_axes)
                for axis, len in zip(layer.axes, shape):
                    axis.length =len
                value = affine(value, activation=activation, x_axes=last_axes, axes=layer.axes, **kargs)
                last_axes = value.axes
        layers.next()
        value = affine(value, activation=activation, x_axes=last_axes, axes=axes, **kargs)
    return value

def L2(x):
    return be.dot(x,x)

class MyTest(be.Model):
    def __init__(self, **kargs):
        super(MyTest, self).__init__(**kargs)

        uni = Uniform(-.01, .01)

        g = self.graph

        g.C = be.Axis()
        g.H = be.Axis()
        g.W = be.Axis()
        g.N = be.Axis()
        g.Y = be.Axis()

        g.x = be.input(axes=(g.C, g.H, g.W, g.N))
        g.y = be.input(axes=(g.Y, g.N))

        layers=[((g.H, g.W), [(32,32)]*2+[(16,16)])]

        g.value = mlp(g.x, activation=be.tanh, x_axes=g.x.axes, shape_spec=layers, axes=g.y.axes, batch_axes=(g.N,), init=uni)

        # L2 regularizer of parameters
        reg = None
        for param in g.value.parameters():
            l2 = L2(param)
            if reg is None:
                reg = l2
            else:
                reg = reg+l2
        g.error = L2(g.y - g.value) + .01*reg

    @be.with_graph_context
    @be.with_environment
    def dump(self):
        for _ in be.get_all_defs():
            print(_)

MyTest().dump()