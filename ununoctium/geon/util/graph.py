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
from builtins import object


class Digraph(object):
    """
    Base class for Directed graph.
    Includes Graphviz visualization, DFS, topsort
    """

    def _graphviz(self, name=''):
        """
        Export the current Digraph to Graphviz

        Args:
            name (str): Name of the resulting graph

        Returns:
            pygraphviz object
        """

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
        """
        Returns the invert of the given adjacency dict (e.g., successors to predecessors)
        """
        result = {x: set() for x in list(adjacency.keys())}
        for x, others in list(adjacency.items()):
            for y in others:
                result[y].add(x)
        return result

    def __init__(self, successors):
        """
        Initialize directed graph from successors dict

        Args:
            successors (dict: op => set(op)): dict that map each op to all its users
        """
        self.successors = successors

    def render(self, fpath, view=True):
        """
        Renders to a graphviz file

        Args:
            fpath (str): file to write too
        """
        self._graphviz().render(fpath, view=view)

    def view(self):
        """ View the graph. Requires pygraphviz """
        self._graphviz().view()

    def dfs(self, fun):
        """
        Performs DFS, applying the provided function to each node

        Args:
            fun (Function): Function to apply to each visited node
        """
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
        for x in sorted(self.inputs, key=lambda x: x.id):
            visit(x, fun)

    @property
    def inputs(self):
        predecessors = Digraph._invert(self.successors)
        return [u for u, vs in iter(list(predecessors.items())) if len(vs) == 0]

    def topsort(self):
        """
        Topological sort of the nodes

        Returns:
            Sorted list of nodes
        """
        result = []
        self.dfs(lambda x: result.insert(0, x))
        return result


class UndirectedGraph(object):
    """
    Base class for Undirected graph.
    Includes Graphviz visualization
    """

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
