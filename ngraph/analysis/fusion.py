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

from ngraph.analysis.dataflow import DataFlowGraph
from ngraph.util.graph import Digraph
import ngraph as ng

try:
    import graphviz
except:
    pass


# Fusion Policies
def never_fusible(op1, op2):
    """
    Default fusion policies: things are not fusible.

    Arguments:
      op1: TODO
      op2: TODO

    Returns:
      Boolean: False
    """

    return False


def gpu_fusible(transformer, op1, op2):
    """
    Fusion policies for the GPU

    Arguments:
      transformer: TODO
      op1: TODO
      op2: TODO

    Returns:
      Boolean
    """

    # Only TensorOps can be merged
    if not isinstance(op1, ng.TensorOp) or not isinstance(op2, ng.TensorOp):
        return False

    shapes1 = op1.tensor_description().shape
    shapes2 = op2.tensor_description().shape
    # Elementwise functions can be merged together if they have the same shapes
    if isinstance(op1, ng.ElementWise) and isinstance(op2, ng.ElementWise) and shapes1 == shapes2:
        return True

    # Reduction following elementwises can be merged
    if isinstance(op1, ng.ElementWise) and isinstance(op2, ng.ReductionOp):
        pre_reduction_shape = op2.args[0].tensor_description().shape
        if pre_reduction_shape == shapes1 or shapes1 == shapes2:
            return True

    # Elementwise following reductions can be merged
    if isinstance(op1, ng.ReductionOp) and isinstance(op2, ng.ElementWise):
        pre_reduction_shape = op1.args[0].tensor_description().shape
        if pre_reduction_shape == shapes2 or shapes1 == shapes2:
            return True

    # Everything else cannot be merged
    return False


class KernelFlowGraph(DataFlowGraph):
    """Class representing a fused dataflow graph"""

    def __init__(self, dataflow, fusible=never_fusible):
        """
        Performs fusion on the provided dataflow graph

        Implementation of: *Fast Greedy Weighted Fusion*, Ken Kennedy,
        Internal journal of Parallel Programming (2002):
        Download: https://drive.google.com/open?id=0B8aziUAQFjRTbDNjeGM5elpFeEk
        """

        # Extracts clusters
        super(KernelFlowGraph, self).__init__(dataflow.transformer,
                                              dataflow.results)
        self.fusible = lambda x, y: fusible(self.transformer, x, y)
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
        clusters = {x: ng.Function(y) if x.device_op else x for x, y in list(clusters.items())}
        self.successors = {
            clusters[a]: {clusters[b] for b in lst} for a, lst in list(
                successors.items())}
        # Saves dataflow for visualization
        self.dataflow = dataflow

    def _graphviz(self, name=''):
        """
        Export fused dataflow to graphviz.
        Involves some hackery to get graphviz to draw edge between subgraphs

        Arguments:
          name (str): Name of the resulting graph

        Returns:
          pygraphviz object
        """

        predecessors = Digraph._invert(self.successors)
        dot = graphviz.Digraph(name, graph_attr={'compound': 'true',
                                                 'nodesep': '.5',
                                                 'ranksep': '.5'})
        leaves = {x for x, y in list(predecessors.items()) if len(y) == 0}
        subgs = {x: x.ops._graphviz('cluster_{}'.format(x.id))
                 for x in self.successors if isinstance(x, ng.Function)}
        # Subgraphs
        for x, sg in list(subgs.items()):
            sg.body.append('color=gray')
            sg.body.append('label={}'.format(x.id))
            dot.subgraph(sg)
        for x in leaves:
            dot.node(x.id, x.graph_label, x.style)
        # Edges
        edges = {(a, b) for a, _ in list(self.successors.items()) for b in _}
        sorts = {x: x.ops.topsort() for x in self.successors if
                 isinstance(x, ng.Function)}
        firsts = {x: sorts[x][0] if isinstance(x, ng.Function) else x for x in
                  self.successors}
        lasts = {x: sorts[x][-1] if isinstance(x, ng.Function) else x for x in
                 self.successors}
        for a, b in edges:
            kw = {}
            if isinstance(a, ng.Function):
                kw['ltail'] = 'cluster_{}'.format(a.name)
            if isinstance(b, ng.Function):
                kw['lhead'] = 'cluster_{}'.format(b.name)
            dot.edge(lasts[a].id, firsts[b].id, **kw)
        return dot

    def _compute_paths(self):
        """
        Computes useful data structures for fusion analysis.

        path_from: maps node v to nodes that have a path from w
        bad_path_from: map node v to nodes that have a bad path from w

        'bad_paths' are paths that can not be merged.

        Returns:
          TODO
        """
        path_from, bad_path_from = dict(), dict()
        reduction_paths = dict()

        order = self.topsort()
        for v in reversed(order):
            path_from[v] = {v}
            bad_path_from[v] = set()

            if isinstance(v, ng.ReductionOp):
                reduction = v.reduction_axes
            else:
                reduction = None

            for w in self.successors[v]:
                path_from[v] |= path_from[w]

                if self.fusible(v, w):
                    for reachable in path_from[w]:
                        path = (reachable, w)
                        new_path = (reachable, v)

                        if path in reduction_paths:
                            if reduction is not None and reduction_paths[path] != reduction:
                                bad_path_from[v] |= {reachable}
                            else:
                                reduction_paths[new_path] = reduction_paths[path]
                        elif reduction is not None:
                            reduction_paths[new_path] = reduction

                    bad_path_from[v] |= bad_path_from[w]
                else:
                    bad_path_from[v] |= path_from[w]
        return path_from, bad_path_from

    def between(self, v, w, path_from):
        """
        Finds all the nodes on any path between v and w.

        Arguments:
          v (operation): start node
          w (operation): end_node
          path_from: (dict): maps node v to nodes that have a path from w

        Returns:
          set of vertices
        """

        vertices = set()
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
        """
        Transfers edges from a node into another

        Arguments:
          v: (operation): node that receives edges
          w: (operation): node that loses edges
          dct: TODO
        """

        dct[v] |= dct.pop(w, set()) - {v}
        for node, connected in list(dct.items()):
            if w in connected:
                connected.remove(w)
                if node != v:
                    connected.add(v)
