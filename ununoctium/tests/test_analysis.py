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
'''
Test the graph analysis functions

'''
from __future__ import print_function

from builtins import range, zip

import geon.backends.graph.analysis as an
import geon.backends.graph.axis as ax
import geon.backends.graph.funs as be


def build_graphs(L, BS):
    with be.bound_environment():
        # Axes
        L = [ax.AxisVar(length=N, name='L%d' % i) for i, N in enumerate(L)]
        BS = ax.AxisVar(length=BS, name='BS')

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
        dfg = an.DataFlowGraph(dW)
        ifg = an.InterferenceGraph(dfg.liveness())
        return dfg, ifg


dataflow, interference = build_graphs([1024, 256, 512, 10], 320)


def test_coloring():
    an.color(interference)
    for u, vs in iter(list(interference.neighbors.items())):
        # u must have a different color from all its neighbors
        for v in vs:
            assert(u.tensor_axes_info.buffer.color != v.tensor_axes_info.buffer.color)
    print('pass coloring')


def test_topsort():
    edges = {(u, v) for u, vs in iter(list(dataflow.successors.items())) for v in vs}
    order = dataflow.topsort()
    for u, v in edges:
        assert(order.index(u) < order.index(v))
    print('pass topsort')


test_topsort()
test_coloring()
