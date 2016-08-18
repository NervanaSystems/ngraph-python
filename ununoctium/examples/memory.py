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
from __future__ import print_function
import geon as be
import numpy as np

have_mxnet = False
try:
    import mxnet as mx
    import mxnet.symbol as sym
    have_mxnet = True
except:
    pass


# TODO If this does something useful, figure out what it is trying to do and fix it,
# otherwise remove the file.
class GraphitiMLP(be.Model):
    """TODO."""

    def __init__(self, L, BS, bprop=True, **kwargs):
        super(GraphitiMLP, self).__init__(**kwargs)

        # Axes
        L = [be.AxisVar(length=N, name='L%d' % i) for i, N in enumerate(L)]
        BS = be.AxisVar(length=BS, name='BS')

        # Builds Network
        activations = [be.tanh for i in range(len(L) - 2)] + [be.softmax]
        X = be.placeholder(axes=(L[0], BS), name='X')
        Y = be.placeholder(axes=(L[-1],), name='Y')
        W = [be.Variable(axes=(L_np1, L_n), name='W%d' % i)
             for i, (L_np1, L_n) in enumerate(zip(L[1:], L[:-1]))]
        A = []
        for i, f in enumerate(activations):
            Aim1 = A[i - 1] if i > 0 else X
            A.append(f(be.dot(W[i], Aim1)))
        Error = be.cross_entropy_multi(A[-1], Y)
        dW = [be.deriv(Error, w) for w in W]

        # Fusion analysis
        results = dW if bprop else [Error]
        transformer = be.NumPyTransformer(fusion=None)
        comp = transformer.computation(results)
        transformer.allocate()
        self.memory = transformer.memory
        transformer.dataflow.view()
        comp()


if have_mxnet:
    class MXNetMLP:
        """TODO."""

        def __init__(self, L, BS, bprop=True, **kwargs):
            # Builds Network
            # activations = ['tanh' for i in range(len(L) - 2)]
            X = sym.Variable('X', shape=(BS, L[0]))
            Y = sym.Variable('Y', shape=(BS,))

            fc, act = [], [X]
            for i, nhid in enumerate(L[1:]):
                fc.append(sym.FullyConnected(data=act[-1], num_hidden=nhid))
                if i == len(L) - 2:
                    act.append(sym.Activation(data=fc[-1], act_type='relu'))
                else:
                    act.append(sym.SoftmaxOutput(
                        data=fc[-1], label=Y, name='softmax'))
            net = act[-1]
            plan = net.simple_bind(
                ctx=mx.cpu(), grad_req='write' if bprop else 'null')

            # Memory internally allocated by MXNet
            # Casted to int internally (rounded down)
            # in average ~.5 smaller than the truth
            bias = .5
            self.memory = (int(plan.debug_str().split(
                '\n')[-3].split()[1]) + bias) * 1024**2
            # Memory required by arguments
            args = plan.arg_arrays
            if plan.grad_arrays:
                args += plan.grad_arrays
            for x in args:
                self.memory += np.prod(x.shape) * 4


layers = [1024, 200, 10]
batch = 320000
bprop = True

graphiti = GraphitiMLP(layers, batch, bprop)

if have_mxnet:
    mxnet = MXNetMLP(layers, batch, bprop)

print('Graphiti: {:.2f} MiB'.format(graphiti.memory * 1024**-2))

if have_mxnet:
    print('MXNet:    {:.2f} MiB (+- 0.5)'.format(mxnet.memory * 1024**-2))
