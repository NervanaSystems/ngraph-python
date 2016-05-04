from neon.util.argparser import NeonArgparser
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks
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


@be.with_name_context
def linear(params, x, x_axes, axes, batch_axes=(), init=None):
    print('x:{x} a:{a} b:{b}'.format(x=x_axes, a=axes, b=batch_axes))
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

        a = self.a
        g = self.naming
        g.x = be.input()
        g.y = be.input()

        layers=[((a.H, a.W), [(14,14)]*2+[(10,10)])]

        self.value = mlp(self.naming.x, activation=be.tanh, x_axes=self.naming.x.axes, shape_spec=layers, axes=self.naming.y.axes, batch_axes=(a.N,))

        self.error = L2(self.naming.y-self.value)

    def dump(self):
        with be.bound_environment(graph=self) as environment:
            a = self.a
            x_axes = (a.H, a.W, a.C, a.N)
            y_axes = (a.Y, a.N)

            environment.set_cached_node_axes(self.naming.x, x_axes)
            environment.set_cached_node_axes(self.naming.y, y_axes)

            print(environment.get_node_axes(self.value))
            print(environment.get_node_axes(self.error))

            print(self.naming.mlp.L[0].linear.weights)
            for layer in self.naming.mlp.L:
                try:
                    print(layer.axes)
                except:
                    pass

            gnp = evaluation.GenNumPy(environment=environment, results=(self.error, self.value))

            x = graph.ArrayWithAxes(np.empty((3, 32, 32, 128)), x_axes)
            y = graph.ArrayWithAxes(np.empty((1000, 128)), y_axes)
            gnp.set_input(self.naming.x, x)
            gnp.set_input(self.naming.y, y)
            gnp.evaluate()

    def train(self):
        # setup data provider
        imgset_options = dict(inner_size=32, scale_range=40, aspect_ratio=110,
                              repo_dir=args.data_dir, subset_pct=args.subset_pct)

        train = ImageLoader(set_name='train', shuffle=True, **imgset_options)

        with be.bound_environment(graph=self) as environment:
            a = self.a
            x_axes = (a.C, a.H, a.W, a.N)
            y_axes = (a.Y, a.N)

            environment.set_cached_node_axes(self.naming.x, x_axes)
            environment.set_cached_node_axes(self.naming.y, y_axes)

            print(environment.get_node_axes(self.value))
            print(environment.get_node_axes(self.error))

            print(self.naming.mlp.L[0].linear.weights)
            for layer in self.naming.mlp.L:
                try:
                    print(layer.axes)
                except:
                    pass

            enp = evaluation.NumPyEvaluator(environment=environment, results=(self.error, self.value))
            for mb_idx, (xraw, yraw) in enumerate(train):
                x = graph.ArrayWithAxes(xraw.array.reshape(3, 32, 32, 128), x_axes)
                y = graph.ArrayWithAxes(yraw.array.reshape(1000, 128), y_axes)
                enp.set_input(self.naming.x, x)
                enp.set_input(self.naming.y, y)

                if mb_idx % 100 == 0:
                    print mb_idx

                vals = enp.evaluate()
                print(vals)
                break

y = MyTest()
#y.dump()
y.train()
