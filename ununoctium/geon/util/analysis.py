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
from builtins import object, range, zip
from collections import defaultdict
from geon.op_graph.op_graph import ComputationOp, AllocationOp, ElementWise, Function, \
    Buffer, ReductionOp, NumPyTensor, Container
from operator import mul
from itertools import combinations
from functools import reduce


class Digraph(object):

    def _graphviz(self, name=''):
        from graphviz import Digraph
        dot = Digraph(name)
        for node, nexts in list(self.successors.items()):
            dot.node(node.id, node.graph_label, node.style)
            for next in nexts:
                dot.node(next.id, next.graph_label, next.style)
                dot.edge(node.id, next.id)
        return dot

    @staticmethod
    def _invert(adjacency):
        result = {x: set() for x in list(adjacency.keys())}
        for x, others in list(adjacency.items()):
            for y in others:
                result[y].add(x)
        return result

    def __init__(self, successors):
        self.successors = successors

    def render(self, fpath, view=True):
        self._graphviz().render(fpath, view=view)

    def view(self):
        self._graphviz().view()

    def dfs(self, fun):
        predecessors = Digraph._invert(self.successors)
        visited = set()
        # Visit single node

        def visit(u, fun):
            if u not in visited:
                vs = self.successors[u]
                for v in sorted(vs, key=lambda x: x.id):
                    if v not in visited:
                        visit(v, fun)
                fun(u)
                visited.add(u)
        # Get output nodes
        inputs = [u for u, vs in iter(list(predecessors.items())) if len(vs) == 0]
        for x in sorted(inputs, key=lambda x: x.id):
            visit(x, fun)

    def topsort(self):
        result = []
        self.dfs(lambda x: result.insert(0, x))
        return result


class DataFlowGraph(Digraph):

    def _fill_successors(self, outputs):
        for w in outputs:
            self.successors[w] |= set()
            for v in w.args:
                self.successors[v].add(w)
                self._fill_successors({v})

    def __init__(self, outputs):
        super(DataFlowGraph, self).__init__(defaultdict(set))
        self._fill_successors(outputs)
        self.outputs = outputs

    def liveness(self):
        can_do_inplace = lambda x: False
        order = self.instructions
        # Initialize
        liveness = dict((op, set()) for op in order)
        keeps = {x.tensor_axes_info.tensor_description for x in self.successors if isinstance(
            x, AllocationOp) and x.tensor_axes_info.read_only}
        liveness[order[-1]] = {x.tensor_axes_info.tensor_description for x in self.outputs} | keeps
        # Update
        for current, previous in reversed(list(zip(order[1:], order[:-1]))):
            use = {x.tensor_axes_info.tensor_description for x in current.args}
            defs = {x.tensor_axes_info.tensor_description for x in current.defs}
            liveness[previous] = use | (liveness[current] - defs)
        # Inplace not possible
        for op in order:
            if not can_do_inplace(op):
                liveness[op] |= {x.tensor_axes_info.tensor_description for x in op.args}

        # print max([sum(map(lambda x: reduce(mul, x.shapes, 1)*x.dtype.itemsize,
        # l)) for l in liveness.itervalues()])*1024**-2
        return liveness

    @property
    def instructions(self):
        return self.topsort()


def never_fusible(op1, op2):
    return isinstance(op1, Container) or isinstance(op2, Container)


def gpu_fusible(op1, op2):
    # Only computations can be merged
    if not isinstance(op1, ComputationOp) or not isinstance(op2, ComputationOp):
        return False

    shapes1, shapes2 = op1.tensor_axes_info.shapes, op2.tensor_axes_info.shapes
    # Elementwise functions can be merged together if they have the same shapes
    if isinstance(op1, ElementWise) and isinstance(op2, ElementWise) and shapes1 == shapes2:
        return True

    # Reduction following elementwises can be merged
    if isinstance(op1, ElementWise) and isinstance(op2, ReductionOp):
        return True

    # Elementwise following reductions can be merged
    if isinstance(op1, ReductionOp) and isinstance(op2, ElementWise):
        return True

    # Everything else cannot be merged
    return False


class KernelFlowGraph(DataFlowGraph):

    def _graphviz(self, name=''):
        predecessors = Digraph._invert(self.successors)
        from graphviz import Digraph as gvDigraph
        dot = gvDigraph(name, graph_attr={
                        'compound': 'true', 'nodesep': '.5', 'ranksep': '.5'})
        leaves = {x for x, y in list(predecessors.items()) if len(y) == 0}
        subgs = {x: x.ops._graphviz('cluster_{}'.format(x.id))
                 for x in self.successors if isinstance(x, Function)}
        # Subgraphs
        for x, sg in list(subgs.items()):
            sg.body.append('color=gray')
            sg.body.append('label={}'.format(x.id))
            dot.subgraph(sg)
        for x in leaves:
            dot.node(x.id, x.graph_label, x.style)
        # Edges
        edges = {(a, b) for a, _ in list(self.successors.items()) for b in _}
        sorts = {x: x.ops.topsort() for x in self.successors if isinstance(x, Function)}
        firsts = {x: sorts[x][0] if isinstance(x, Function) else x for x in self.successors}
        lasts = {x: sorts[x][-1] if isinstance(x, Function) else x for x in self.successors}
        for a, b in edges:
            kw = {}
            if isinstance(a, Function):
                kw['ltail'] = 'cluster_{}'.format(a.id)
            if isinstance(b, Function):
                kw['lhead'] = 'cluster_{}'.format(b.id)
            dot.edge(lasts[a].id, firsts[b].id, **kw)
        return dot

    def _compute_paths(self):
        path_from, bad_path_from = dict(), dict()
        order = self.topsort()
        for v in reversed(order):
            path_from[v] = {v}
            bad_path_from[v] = set()
            for w in self.successors[v]:
                path_from[v] |= path_from[w]
                if self.fusible(v, w):
                    bad_path_from[v] |= bad_path_from[w]
                else:
                    bad_path_from[v] |= path_from[w]
        return path_from, bad_path_from

    def between(self, v, w, path_from):
        vertices = set()
        # Initialize worklists to all successors of v who can reach w
        worklist = {w}
        worklist |= {x for x in self.successors[v] if w in path_from[x]}
        while worklist:
            # Update worklist
            x = worklist.pop()
            if x != w:
                worklist |= {y for y in self.successors[
                    x] if w in path_from[y]}
            # Add vertices
            vertices |= {x}
        return vertices

    def transfer_edges(self, v, w, dct):
        dct[v] |= dct.pop(w, set()) - {v}
        for node, connected in list(dct.items()):
            if w in connected:
                connected.remove(w)
                if node != v:
                    connected.add(v)

    def __init__(self, dataflow, fusible=never_fusible):
        # Extracts clusters
        self.fusible = fusible
        super(KernelFlowGraph, self).__init__(dataflow.outputs)
        successors = self.successors
        path_from, bad_path_from = self._compute_paths()
        edges = {(a, b) for a, _ in successors.items() for b in _}
        edges = sorted(edges, key=lambda x: (x[0].id, x[1].id))
        clusters = dict((x, {x}) for e in edges for x in e)
        while edges:
            # Pop edges and adjusts order if necessary
            v, w = edges.pop()
            # Cannot be fused
            if w in bad_path_from[v]:
                continue
            # Merge vertices between v and w
            to_merge = self.between(v, w, path_from)
            for x in to_merge:
                clusters[v] |= clusters.pop(x)
                self.transfer_edges(v, x, successors)
                self.transfer_edges(v, x, path_from)
                self.transfer_edges(v, x, bad_path_from)
            edges = {(a, b) for a, _ in successors.items() for b in _}
            edges = sorted(edges, key=lambda x: (x[0].id, x[1].id))
        # Creates adjacency list for each cluster
        extract_subgraph = lambda R: dict(
            (a, b & R) for a, b in list(dataflow.successors.items()) if a in R)
        clusters = {x: extract_subgraph(y) for x, y in list(clusters.items())}
        # Creates final adjacency list
        clusters = {x: Function(y) if isinstance(
            x, ComputationOp) else x for x, y in list(clusters.items())}
        self.successors = {
            clusters[a]: {
                clusters[b] for b in lst} for a,
            lst in list(
                successors.items())}
        # Saves dataflow for visualization
        self.dataflow = dataflow


class UndirectedGraph(object):

    def __init__(self, neighbors):
        self.neighbors = neighbors

    def _graphviz(self, name=''):
        from graphviz import Graph
        dot = Graph()
        processed = set()
        for na, _ in list(self.neighbors.items()):
            dot.node(na.id, na.graph_label, na.style)
            for nb in _:
                dot.node(nb.id, nb.graph_label, nb.style)
                if (nb, na) not in processed:
                    dot.edge(na.id, nb.id)
                    processed.add((na, nb))
        return dot

    def render(self, fpath, view=True):
        self._graphviz().render(fpath, view=view)

    def view(self):
        self._graphviz().view()


class InterferenceGraph(UndirectedGraph):

    def __init__(self, lives):
        neighbors = {x: set() for l in list(lives.values()) for x in l}
        edges = [(u, v) for l in list(lives.values()) for u, v in combinations(l, 2)]
        for u, v in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)
        super(InterferenceGraph, self).__init__(neighbors)
        self.weights = {x: max(1, reduce(mul, x.shape, 1)) *
                        x.dtype.itemsize for x in neighbors}

    def color(self):
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
        return total_mem, buffers


def _random_colors(N, alpha=.5):
    from colorsys import hsv_to_rgb
    HSV = [[x * 1.0 / N, 0.5, 0.5] for x in range(N)]
    RGBA = [x + (alpha,) for x in [hsv_to_rgb(*x) for x in HSV]]
    RGBA = [[int(y * 255) for y in x] for x in RGBA]
    HEX = ["#{:02x}{:02x}{:02x}{:02x}".format(
        r, g, b, a) for r, g, b, a in RGBA]
    return HEX


def assign_buffers(outputs, fusible=None):
    dfg = DataFlowGraph(outputs)
    if fusible:
        dfg = KernelFlowGraph(dfg, fusible)
    ifg = InterferenceGraph(dfg.liveness())
    memory, buffers = ifg.color()
    # Binds initializers
    for op in dfg.successors:
        buffer = op.tensor_axes_info.tensor_description.buffer
        for i in op.initializers:
            i.tensor_axes_info.tensor_description.buffer = buffer
            for a in i.args:
                if isinstance(a, NumPyTensor):
                    a.tensor_axes_info.tensor_description.buffer = Buffer(-1, a.nptensor.size)
                    a.tensor_axes_info.tensor_description.buffer.data = a.nptensor
    # set style
    cmap = _random_colors(len(buffers), .5)
    for op in dfg.successors:
        op.style = {'style': 'filled', 'fillcolor': cmap[
            op.tensor_axes_info.tensor_description.buffer.color]}
    # dfg.view()
    return dfg, memory
