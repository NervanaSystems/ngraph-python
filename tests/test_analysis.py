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
"""
Test the graph analysis functions.
"""

from __future__ import print_function

import ngraph as ng
import ngraph.transformers as ngt
import ngraph.analysis as an
from builtins import range, zip


def build_graphs(L, BS):
    """
    TODO.

    Arguments:
      L: TODO
      BS: TODO

    Returns:
      TODO
    """
    # Axes
    L = [ng.make_axis(length=N, name='L%d' % i) for i, N in enumerate(L)]
    BS = ng.make_axis(length=BS, name='BS')

    # Builds Network
    activations = [ng.tanh for i in range(len(L) - 2)] + [ng.softmax]
    X = ng.placeholder((L[0], BS)).named('X')
    Y = ng.placeholder((L[-1],)).named('Y')
    W = [ng.variable((L_np1, L_n - 1)).named('W%d' % i)
         for i, (L_np1, L_n) in enumerate(zip(L[1:], L[:-1]))]
    A = []
    for i, f in enumerate(activations):
        Aim1 = A[i - 1] if i > 0 else X
        A.append(f(ng.dot(W[i], Aim1)))
    Error = ng.cross_entropy_multi(A[-1], Y)
    dW = [ng.deriv(Error, w) for w in W]
    transformer = ngt.make_transformer()
    dfg = an.DataFlowGraph(transformer, dW)
    ifg = an.InterferenceGraph(dfg.liveness())
    return dfg, ifg


dataflow, interference = build_graphs([1024, 256, 512, 10], 320)


def test_coloring():
    """TODO."""
    interference.color()
    for u, vs in iter(interference.neighbors.items()):
        # u must have a different color from all its neighbors
        for v in vs:
            assert(u.buffer.color != v.buffer.color)
    print('pass coloring')


def test_topsort():
    """TODO."""
    edges = {(u, v) for u, vs in iter(list(dataflow.successors.items())) for v in vs}
    order = dataflow.topsort()
    for u, v in edges:
        assert(order.index(u) < order.index(v))
    print('pass topsort')
