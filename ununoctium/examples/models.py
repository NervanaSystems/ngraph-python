from __future__ import division, print_function
from builtins import range, zip
from geon.backends.graph.graphneon import *

import geon.backends.graph.graph as graph
import geon.backends.graph.pycudatransform as evaluation
import numpy as np

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.set_defaults(backend='dataloader')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()

@be.with_name_scope
def linear(ns, x, axes, init=None, bias=None):
    ns.weights = be.Variable(axes=be.linear_map_axes(be.sample_axes(x.axes), be.sample_axes(axes)), init=init)
    result = be.dot(ns.weights, x)
    if bias is not None:
        ns.bias = be.Variable(axes=be.sample_axes(result), init=bias)
        result = result + ns.bias
    return result


def affine(x, activation, **kargs):
    return activation(linear(x, **kargs))


@be.with_name_scope
def mlp(ns, x, activation, shape_spec, axes, **kargs):
    value = x
    with be.name_scope_list('L') as name_scopes:
        for hidden_activation, hidden_axes, hidden_shapes in shape_spec:
            for shape in hidden_shapes:
                with be.next_name_scope(name_scopes) as nns:
                    nns.axes = tuple(be.AxisVar(like=axis, length=length) for axis, length in zip(hidden_axes, shape))
                    value = affine(value, activation=hidden_activation, axes=nns.axes, **kargs)
        with be.next_name_scope(name_scopes):
            value = affine(value, activation=activation, axes=axes, **kargs)
    return value


def grad_descent(cost):
    learning_rate = be.placeholder(axes=())
    params = cost.parameters()
    derivs = [be.deriv(cost, param) for param in params]
    updates = be.doall(all=[be.assign(param, param - learning_rate * deriv) for param, deriv in zip(params, derivs)])
    return learning_rate, updates


class MyTest(be.Model):
    def __init__(self, **kargs):
        super(MyTest, self).__init__(**kargs)

        uni = Uniform(-.001, .001)

        g = self.graph

        be.set_batch_axes([ax.N])
        be.set_phase_axes([ax.Phi])

        g.x = be.placeholder(axes=(ax.C, ax.H, ax.W, ax.N))
        g.y = be.placeholder(axes=(ax.Y, ax.N))

        #layers = [(be.tanh, (g.H, g.W), [(16, 16)] * 1 + [(4, 4)])]
        layers = [(be.tanh, (ax.Y,), [(200,)])]

        g.value = mlp(g.x, activation=be.softmax, shape_spec=layers, axes=g.y.axes, init=uni)

        g.error = be.cross_entropy_multi(g.value, g.y)

        # L2 regularizer of parameters
        reg = None
        for param in g.value.parameters():
            l2 = L2(param)
            if reg is None:
                reg = l2
            else:
                reg = reg + l2

        g.loss = g.error + .01 * reg

    @be.with_graph_scope
    def dump(self):
        g = self.graph

        g.x.value = np.empty((3, 32, 32, 128))
        g.y.value = np.empty((1000, 128))

        learning_rate = be.placeholder(axes=())
        params = g.error.parameters()
        derivs = [be.deriv(g.error, param) for param in params]

        gnp = evaluation.GenNumPy(results=derivs)
        gnp.evaluate()

    @be.with_graph_scope
    def train(self):
        with be.bound_environment() as env:
            # setup data provider
            imgset_options = dict(inner_size=32, scale_range=40, aspect_ratio=110,
                                  repo_dir=args.data_dir, subset_pct=args.subset_pct)

            train = ImageLoader(set_name='train', shuffle=True, **imgset_options)
            #train = ImageLoader(set_name='train', shuffle=False, do_transforms=False, **imgset_options)
            test = ImageLoader(set_name='validation', shuffle=False, do_transforms=False, **imgset_options)

            graph = self.graph
            ax.N.length = train.bsz
            c, h, w = train.shape
            ax.C.length = c
            ax.H.length = h
            ax.W.length = w
            ax.Y.length = train.nclasses

            learning_rate, updates = grad_descent(graph.error)

            enp = be.NumPyTransformer(results=[self.graph.value, graph.error, updates])

            for epoch in range(args.epochs):
                print("Epoch {epoch}".format(epoch=epoch))
                training_error = 0
                training_n = 0
                learning_rate.value = .1 / (1 + epoch) / train.bsz
                for mb_idx, (xraw, yraw) in enumerate(train):
                    graph.x.value = xraw
                    graph.y.value = yraw
                    vals = enp.evaluate()
                    training_error += vals[graph.error] / train.bsz
                    training_n += 1
                    # break

                print('Training error: {e}'.format(e=training_error/training_n))
                self.test(env, test)

                train.reset()

            return env

    @be.with_graph_scope
    def test(self, env, test):
        graph = self.graph
        with be.bound_environment(env):
            enp = be.NumPyTransformer(results=[self.graph.value, graph.error])
            total_error = 0
            n = 0
            for mb_idx, (xraw, yraw) in enumerate(test):
                graph.x.value = xraw
                graph.y.value = yraw
                vals = enp.evaluate()
                total_error += vals[graph.error] / test.bsz
                n += 1
                # break
            print("Test error: {e}".format(e=total_error / n))


y = MyTest()
#y.dump()
env = y.train()
#y.test(env)
