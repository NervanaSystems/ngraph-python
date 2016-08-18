#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from __future__ import division, print_function
from builtins import range, zip
from geon.frontends.neon import *  # noqa

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.set_defaults(backend='dataloader')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()


@be.with_name_scope
def linear(ns, x, axes, init=None, bias=None):
    """
    TODO.

    Arguments:
      ns: TODO
      x: TODO
      axes: TODO
      init: TODO
      bias: TODO

    Returns:
      TODO
    """
    ns.weights = be.Variable(
        axes=be.linear_map_axes(
            be.sample_axes(x.axes),
            be.sample_axes(axes)),
        init=init)
    result = be.dot(ns.weights, x)
    if bias is not None:
        ns.bias = be.Variable(axes=be.sample_axes(result), init=bias)
        result = result + ns.bias
    return result


def affine(x, activation, **kwargs):
    """
    TODO.

    Arguments:
      x: TODO
      activation: TODO
      **kwargs: TODO

    Returns:
      TODO
    """
    return activation(linear(x, **kwargs))


@be.with_name_scope
def mlp(ns, x, activation, shape_spec, axes, **kwargs):
    """
    TODO.

    Arguments:
      ns: TODO
      x: TODO
      activation: TODO
      shape_spec: TODO
      axes: TODO
      **kwargs: TODO

    Returns:
      TODO
    """
    value = x
    with be.name_scope_list('L') as name_scopes:
        for hidden_activation, hidden_axes, hidden_shapes in shape_spec:
            for shape in hidden_shapes:
                with be.next_name_scope(name_scopes) as nns:
                    nns.axes = tuple(be.AxisVar(length=length)
                                     for axis, length in zip(hidden_axes, shape))
                    value = affine(
                        value,
                        activation=hidden_activation,
                        axes=nns.axes,
                        **kwargs)
        with be.next_name_scope(name_scopes):
            value = affine(value, activation=activation, axes=axes, **kwargs)
    return value


def grad_descent(cost):
    """
    TODO.

    Arguments:
      cost: TODO

    Returns:
      TODO
    """
    learning_rate = be.placeholder(axes=(), name='learning_rate')
    variables = cost.variables()
    derivs = [be.deriv(cost, variable) for variable in variables]
    updates = be.doall(all=[
        be.assign(variable, variable - learning_rate * deriv)
        for variable, deriv in zip(variables, derivs)
    ])
    return learning_rate, updates


class MyTest(be.Model):
    """TODO."""

    def __init__(self, **kwargs):
        super(MyTest, self).__init__(**kwargs)

        uni = Uniform(-.001, .001)

        g = self.graph

        be.set_batch_axes([ax.N])
        be.set_phase_axes([ax.Phi])

        g.x = be.placeholder(axes=(ax.C, ax.H, ax.W, ax.N))
        g.y = be.placeholder(axes=(ax.Y, ax.N))

        # layers = [(be.tanh, (g.H, g.W), [(16, 16)] * 1 + [(4, 4)])]
        layers = [(be.tanh, (ax.Y,), [(200,)])]

        g.value = mlp(g.x, activation=be.softmax,
                      shape_spec=layers, axes=g.y.axes, init=uni)

        g.errors = be.cross_entropy_multi(g.value, g.y)
        g.error = be.sum(g.errors, out_axes=())

        # L2 regularizer of variables
        reg = None
        for variable in g.value.variables():
            l2 = L2(variable)
            if reg is None:
                reg = l2
            else:
                reg = reg + l2

        g.loss = g.error + .01 * reg

        self.transformer = be.NumPyTransformer()
        self.train_comp = None
        self.test_comp = None

    @be.with_graph_scope
    def train(self):
        """TODO."""
        # setup data provider
        imgset_options = dict(
            inner_size=32,
            scale_range=40,
            aspect_ratio=110,
            repo_dir=args.data_dir,
            subset_pct=args.subset_pct)

        train = ImageLoader(
            set_name='train', shuffle=True, **imgset_options)
        test = ImageLoader(
            set_name='validation',
            shuffle=False,
            do_transforms=False,
            **imgset_options)

        graph = self.graph
        ax.N.length = train.bsz
        c, h, w = train.shape
        ax.C.length = c
        ax.H.length = h
        ax.W.length = w
        ax.Y.length = train.nclasses

        learning_rate, updates = grad_descent(graph.error)

        self.train_comp = self.transformer.computation(graph.error, graph.x, graph.y, updates)
        self.test_comp = self.transformer.computation(graph.error, graph.x, graph.y)
        self.transformer.allocate()

        for epoch in range(args.epochs):
            print("Epoch {epoch}".format(epoch=epoch))
            training_error = 0
            training_n = 0
            learning_rate.value = .1 / (1 + epoch) / train.bsz
            for mb_idx, (x, y) in enumerate(train):
                error = self.train_comp(x.reshape(graph.x.shape.lengths),
                                        y.reshape(graph.y.shape.lengths))
                training_error += error / train.bsz
                training_n += 1
                # break

            print('Training error: {e}'.format(
                e=training_error / training_n))
            self.test(test)

            train.reset()

    def test(self, test):
        graph = self.graph
        total_error = 0
        n = 0
        for mb_idx, (xraw, yraw) in enumerate(test):
            error = self.test_comp(xraw.reshape(graph.x.shape.lengths),
                                   yraw.reshape(graph.y.shape.lengths))
            total_error += error / test.bsz
            n += 1
            # break
        print("Test error: {e}".format(e=total_error / n))


y = MyTest()
y.train()
