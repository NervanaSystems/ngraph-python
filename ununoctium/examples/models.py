from neon.util.argparser import NeonArgparser
from neon.data import ImageLoader
import geon.backends.graph.dataloaderbackend

import geon.backends.graph.funs as be
import geon.backends.graph.graph as graph
import geon.backends.graph.evaluation as evaluation
import numpy as np

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.set_defaults(backend='dataloader')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()

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
                layer.axes = tuple(axis.like() for axis in hidden_axes)
                for axis, len in zip(layer.axes, shape):
                    axis[len]
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

        a = self.a
        g = self.graph
        g.x = be.input(axes=(a.C, a.H, a.W, a.N))
        g.y = be.input(axes=(a.Y, a.N))

        layers=[((a.H, a.W), [(14,14)]*2+[(10,10)])]

        g.value = mlp(g.x, activation=be.tanh, x_axes=g.x.axes, shape_spec=layers, axes=g.y.axes, batch_axes=(a.N,), init=uni)

        # L2 regularizer of parameters
        reg = None
        for param in g.value.parameters():
            l2 = L2(param)
            if reg is None:
                reg = l2
            else:
                reg = reg+l2

        g.error = L2(g.y - g.value) + reg

    @be.with_graph_context
    @be.with_environment
    def dump(self):
        a = self.a

        self.graph.x.value = be.ArrayWithAxes(np.empty((3, 32, 32, 128)), (a.C, a.H, a.W, a.N))
        self.graph.y.value = be.ArrayWithAxes(np.empty((1000, 128)), (a.Y, a.N))


        gnp = evaluation.GenNumPy(results=(self.graph.error, self.graph.value))
        gnp.evaluate()


    @be.with_graph_context
    @be.with_environment
    def train(self):
        # setup data provider
        imgset_options = dict(inner_size=32, scale_range=40, aspect_ratio=110,
                              repo_dir=args.data_dir, subset_pct=args.subset_pct)

        train = ImageLoader(set_name='train', shuffle=True, **imgset_options)

        a = self.a
        a.N[train.bsz]
        c, h, w = train.shape
        a.C[c]
        a.H[h]
        a.W[w]
        a.Y[train.nclasses]


        a = self.a

        error = self.graph.error
        learning_rate = be.input(axes=())
        params = error.parameters()
        derivs = [be.deriv(error, param) for param in params]

        updates = be.doall(all=[be.decrement(param, learning_rate*deriv) for param, deriv in zip(params, derivs)])

        enp = evaluation.NumPyEvaluator(results=[self.graph.value, error, updates])
        enp.initialize()
        for mb_idx, (xraw, yraw) in enumerate(train):
            self.graph.x.value = be.ArrayWithAxes(xraw.array, shape=(train.shape, train.bsz), axes=(a.C, a.H, a.W, a.N))
            self.graph.y.value = be.ArrayWithAxes(yraw.array, shape=(train.nclasses, train.bsz), axes=(a.Y, a.N))
            learning_rate.value = be.ArrayWithAxes(.01, shape=(), axes=())

            if mb_idx % 100 == 0:
                print mb_idx

            vals = enp.evaluate()
            print(vals)
            break

        print(be.get_current_environment().get_resolved_node_axes(self.graph.value))
        print(be.get_current_environment().get_resolved_node_axes(self.graph.error))


y = MyTest()
#y.dump()
y.train()
