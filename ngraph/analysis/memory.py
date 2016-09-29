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

from __future__ import division
from operator import mul
from functools import reduce
from itertools import combinations
from ngraph.util.graph import UndirectedGraph
from ngraph.analysis.dataflow import DataFlowGraph
from ngraph.analysis.fusion import KernelFlowGraph
from ngraph.op_graph.op_graph import Buffer, TensorOp


def _random_colors(N, alpha=.5):
    """
    Creates a map of N color of transparency alpha.

    Arguments:
      N: TODO
      alpha: TODO

    Returns:
      TODO

    """
    from colorsys import hsv_to_rgb
    HSV = [[x * 1.0 / N, 0.5, 0.5] for x in range(N)]
    RGBA = [x + (alpha,) for x in [hsv_to_rgb(*x) for x in HSV]]
    RGBA = [[int(y * 255) for y in x] for x in RGBA]
    HEX = ["#{:02x}{:02x}{:02x}{:02x}".format(
        r, g, b, a) for r, g, b, a in RGBA]
    return HEX


class InterferenceGraph(UndirectedGraph):
    """
    Interference graph. Undirected graph containing a node for each
    tensor, and an edge between tensors that are live at the same time.

    This class implements an graph coloring algorithm.

    In a standard graph coloring problem you want to minimize the number of
    buffers allocated.  in this variant of the graph coloring problem we want
    to minimize the total buffer space allocated.  In academic literature this
    variant is referred to as ____.
    """

    def __init__(self, lives):
        """
        Creates the interference graph from the provided liveness information.
        There is an edge in the interference graph whenever two variables are
        live at the same time. Each node is weighted by the memory requirement
        of the underlying tensor.

        This seems to be the performance bottleneck for very large graphs.
        Construction could be optimized, or coloring could be done directly
        from the liveness information.

        Arguments:
          lives (op => set(tensor_description)): Live tensors at each point
                                                 Typically the output of dataflow.liveness()
        """
        neighbors = {x: set() for l in list(lives.values()) for x in l}
        edges = [(u, v) for l in list(lives.values()) for u, v in combinations(l, 2)]
        for u, v in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)
        super(InterferenceGraph, self).__init__(neighbors)
        self.weights = {x: max(1, reduce(mul, x.shape, 1)) *
                        x.dtype.itemsize for x in neighbors}

    def color(self):
        """
        Performs weighted graph coloring on this interference graph.
        Basically implements:
        *Buffer allocation in regular dataflow networks:
        an approach based on coloring circular-arc graphs*, R. Govindarajan
        https://drive.google.com/open?id=0B8aziUAQFjRTa2Mzb2VUWEFaRXM
        """

        neighbors = self.neighbors
        weights = self.weights
        partitions = []
        buffers = []
        queue = sorted(weights, key=lambda x: (weights[x], ), reverse=True)
        while queue:
            u = queue.pop(0)
            # Creates a new set and grows it as much as possible
            S = {u}
            N = neighbors[u]
            for x in queue:
                if x not in N:
                    S |= {x}
                    N |= neighbors[x]
            partitions.append(S)
            color = len(partitions) - 1
            buffers.append(Buffer(color, weights[u]))
            # Update remaining nodes
            queue = [x for x in queue if x not in S]
            for s in S:
                s.buffer = buffers[color]
        total_mem = sum([x.size for x in buffers])
        cmap = _random_colors(len(buffers), .5)
        for tensor in neighbors:
            tensor.style = {'style': 'filled', 'fillcolor': cmap[tensor.buffer.color]}
        return total_mem, buffers


def assign_buffers(transformer, results, fusible=None):
    """
    Performs dataflow analysis of the graph defined by the provide results.
    Assigns buffer to each node.

    Arguments:
      transformer: TODO
      fusible: TODO
      results: results to build the graph from

    Returns:
      dfg (DataFlowGraph/KernelFlowGraph): dataflow of the computation
      memory (int): Memory usage of the computations
    """

    dfg = DataFlowGraph(transformer, results)
    all_ops = dfg.successors.keys()
    if fusible:
        dfg = KernelFlowGraph(dfg, fusible)
    ifg = InterferenceGraph(dfg.liveness())
    memory, buffers = ifg.color()
    # set style
    for op in all_ops:
        if isinstance(op, TensorOp):
            tensor = op.tensor_description()
            op.style = tensor.style
    # dfg.view()
    return dfg, memory
