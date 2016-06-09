from neon.util.argparser import NeonArgparser
from neon.data import ImageLoader
import geon.backends.graph.dataloaderbackend
from neon.initializers import Uniform, Constant

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
    params.weights = be.Parameter(axes=axes + x_axes - batch_axes, init=init)
    params.bias = be.Parameter(axes=axes, init=init)
    return be.dot(params.weights, x) + params.bias


def affine(x, activation, batch_axes=None, **kargs):
    return activation(linear(x, batch_axes=batch_axes, **kargs), batch_axes=batch_axes)


@be.with_name_context
def mlp(params, x, activation, x_axes, shape_spec, axes, **kargs):
    value = x
    last_axes = x_axes
    with be.layers_named('L') as layers:
        for hidden_activation, hidden_axes, hidden_shapes in shape_spec:
            for shape in hidden_shapes:
                with be.next_layer(layers) as layer:
                    layer.axes = tuple(be.Axis(like=axis) for axis in hidden_axes)
                    for axis, length in zip(layer.axes, shape):
                        axis.length = length
                    value = affine(value, activation=hidden_activation, x_axes=last_axes, axes=layer.axes, **kargs)
                    last_axes = value.axes
        with be.next_layer(layers):
            value = affine(value, activation=activation, x_axes=last_axes, axes=axes, **kargs)
    return value


def L2(x):
    return be.dot(x, x)


def cross_entropy(y, t):
    """

    :param y:  Estimate
    :param t: Actual 1-hot data
    :return:
    """
    return -be.sum(be.log(y) * t)


class MyTest(be.Model):
    def __init__(self, **kargs):
        super(MyTest, self).__init__(**kargs)

        uni = Uniform(-.001, .001)

        g = self.graph

        g.C = be.Axis()
        g.H = be.Axis()
        g.W = be.Axis()
        g.N = be.Axis()
        g.Y = be.Axis()

        g.x = be.input(axes=(g.C, g.H, g.W, g.N))
        g.y = be.input(axes=(g.Y, g.N))

        #layers = [(be.tanh, (g.H, g.W), [(16, 16)] * 1 + [(4, 4)])]
        layers = [(be.tanh, (g.Y,), [(200,)])]

        g.value = mlp(g.x, activation=be.softmax, x_axes=g.x.axes, shape_spec=layers, axes=g.y.axes, batch_axes=(g.N,),
                      init=uni)

        g.error = cross_entropy(g.value, g.y)

        # L2 regularizer of parameters
        reg = None
        for param in g.value.parameters():
            l2 = L2(param)
            if reg is None:
                reg = l2
            else:
                reg = reg + l2

        g.loss = g.error + .01 * reg


    @be.with_graph_context
    @be.with_environment
    def dump(self):
        g = self.graph

        g.x.value = be.ArrayWithAxes(np.empty((3, 32, 32, 128)), (g.C, g.H, g.W, g.N))
        g.y.value = be.ArrayWithAxes(np.empty((1000, 128)), (g.Y, g.N))

        learning_rate = be.input(axes=())
        params = g.error.parameters()
        derivs = [be.deriv(g.error, param) for param in params]

        gnp = evaluation.GenNumPy(results=derivs)
        gnp.evaluate()

    @be.with_graph_context
    def train(self):
        with be.bound_environment() as env:
            # setup data provider
            imgset_options = dict(inner_size=32, scale_range=40, aspect_ratio=110,
                                  repo_dir=args.data_dir, subset_pct=args.subset_pct)

            train = ImageLoader(set_name='train', shuffle=True, **imgset_options)
            #train = ImageLoader(set_name='train', shuffle=False, do_transforms=False, **imgset_options)
            test = ImageLoader(set_name='validation', shuffle=False, do_transforms=False, **imgset_options)

            g = self.graph
            g.N.length = train.bsz
            c, h, w = train.shape
            g.C.length = c
            g.H.length = h
            g.W.length = w
            g.Y.length = train.nclasses

            learning_rate = be.input(axes=())
            params = g.error.parameters()
            derivs = [be.deriv(g.loss, param) for param in params]

            updates = be.doall(all=[be.decrement(param, learning_rate * deriv) for param, deriv in zip(params, derivs)])

            enp = evaluation.NumPyEvaluator(results=[self.graph.value, g.error, updates]+derivs)
            enp.initialize()

            for epoch in range(args.epochs):
                print("Epoch {epoch}".format(epoch=epoch))
                training_error = 0
                training_n = 0
                learning_rate.value = be.ArrayWithAxes(.001/(1+epoch), shape=(), axes=())
                for mb_idx, (xraw, yraw) in enumerate(train):
                    g.x.value = be.ArrayWithAxes(xraw.array, shape=(train.shape, train.bsz), axes=(g.C, g.H, g.W, g.N))
                    g.y.value = be.ArrayWithAxes(yraw.array, shape=(train.nclasses, train.bsz), axes=(g.Y, g.N))
                    vals = enp.evaluate()
                    training_error += vals[g.error].array/128.0
                    training_n += 1
                    # break

                print('Training error: {e}'.format(e=training_error/training_n))
                self.test(env, test)

                train.reset()

            return env

    @be.with_graph_context
    def test(self, env, test):
        g = self.graph
        with be.bound_environment(env):
            enp = evaluation.NumPyEvaluator(results=[self.graph.value, g.error])
            total_error = 0
            n = 0
            for mb_idx, (xraw, yraw) in enumerate(test):
                g.x.value = be.ArrayWithAxes(xraw.array, shape=(test.shape, test.bsz), axes=(g.C, g.H, g.W, g.N))
                g.y.value = be.ArrayWithAxes(yraw.array, shape=(test.nclasses, test.bsz), axes=(g.Y, g.N))

                vals = enp.evaluate()
                total_error += vals[g.error].array / test.bsz
                n += 1
                # break
            print("Test error: {e}".format(e=total_error/n))


y = MyTest()
#y.dump()
env = y.train()
#y.test(env)
