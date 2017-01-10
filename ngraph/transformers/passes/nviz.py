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
import json
import random
import tempfile

import six

from ngraph.transformers.passes.passes import GraphPass


class JSONPass(GraphPass):
    """
    Example of a graphpass that dumps a JSON version of the graph to disk
    """
    def __init__(self):
        super(JSONPass, self).__init__()

    def do_pass(self, ops, init):
        nodes = dict()
        edges = []

        def get_id(op):
            return op.name

        def add_op(op):
            opid = get_id(op)
            if opid not in nodes:
                op_name = op.__class__.__name__
                filename = op.filename.split('/')[-1]
                axes = ''
                description = op_name
                if hasattr(op, 'axes'):
                    axes = str(op.axes)
                nodes[opid] = dict(id=opid, label=description, op_name=op_name, filename=filename,
                                   lineno=op.lineno, axes=axes, tags=op.metadata)

        frontier = set(ops)
        visited = set()
        while len(frontier) > 0:
            op = frontier.pop()
            add_op(op)
            visited.add(op)
            for arg in op.args:
                if arg not in visited:
                    frontier.add(arg)
                edges.append({'from': get_id(op), 'to': get_id(arg)})
            for arg in op.initializers:
                if arg not in visited:
                    frontier.add(arg)
                edges.append({'from': get_id(op), 'to': get_id(arg), 'color': 'green'})
            for arg in op.other_deps:
                if arg not in visited:
                    frontier.add(arg)
                edges.append({'from': get_id(op), 'to': get_id(arg), 'color': 'blue'})

        with tempfile.TemporaryFile('w') as fid:
            json.dump(dict(nodes=nodes.values(), edges=edges), fid)


class VizPass(GraphPass):
    """
    A graph pass that visualizes nervana graphs and displays them to the user
    """
    def __init__(self, subgraph_attr=None, show_axes=False, show_all_metadata=False):
        super(VizPass, self).__init__()
        self.show_axes = show_axes
        self.show_all_metadata = show_all_metadata
        self.subgraph_attr = subgraph_attr

    def get_subgraphs(self, ops):
        clusters = set()
        for op in ops:
            clusters.add(op.metadata.get(self.subgraph_attr, None))
        return clusters

    def random_color(self, alpha=0.2):
        """ Return random color """
        from colorsys import hsv_to_rgb
        HSV = [[random.random(), 0.5, 0.5]]
        RGBA = [x + (alpha,) for x in [hsv_to_rgb(*x) for x in HSV]]
        RGBA = [[int(y * 255) for y in x] for x in RGBA]
        HEX = ["#{:02x}{:02x}{:02x}{:02x}".format(
            r, g, b, a) for r, g, b, a in RGBA]
        return HEX[0]

    def add_op_to_graph(self, op, graph):
        op_label = op.name
        if hasattr(op, 'axes') and self.show_axes:
            op_label += "\n{}".format(op.axes)
        if self.show_all_metadata:
            for k, v in six.iteritems(op.metadata):
                op_label += "\n{}={}".format(k, v)
        graph.node(op.name, op_label)
        for arg in op.args:
            graph.edge(op.name, arg.name)
        for arg in op.initializers:
            graph.edge(op.name, arg.name, color='green')
        for arg in op.other_deps:
            graph.edge(op.name, arg.name, color='blue')
        if op.forwarded and op.forwarded is not op:
            graph.edge(op.name, op.forwarded.name, color='red')

    def do_pass(self, ops, inits):
        try:
            import graphviz
        except ImportError:
            raise ImportError("You tried to use the ShowGraph transformer pass but did "
                              "not have the python graphviz library installed")
        # Get all ops from this set
        frontier = set(ops)
        visited = set()
        while len(frontier) > 0:
            op = frontier.pop()
            visited.add(op)
            for arg in op.args:
                if arg not in visited:
                    frontier.add(arg)
            for arg in op.initializers:
                if arg not in visited:
                    frontier.add(arg)
            for arg in op.other_deps:
                if arg not in visited:
                    frontier.add(arg)

        visited_ops = list(visited)
        vg = graphviz.Digraph(node_attr={'shape': 'box'},
                              graph_attr={'nodesep': '.5',
                                          'ranksep': '.5'})
        if self.subgraph_attr is not None:
            subgraphs = {}
            for subgraph_name in self.get_subgraphs(visited_ops):
                if subgraph_name not in subgraphs and subgraph_name is not None:
                    sg = graphviz.Digraph(name='cluster_{}'.format(subgraph_name))
                    sg.body.append('color="{}"'.format(self.random_color()))
                    sg.body.append('style=filled')
                    sg.body.append('label="{}"'.format(subgraph_name))
                    subgraphs[subgraph_name] = sg
            for op in visited_ops:
                subgraph_name = op.metadata.get(self.subgraph_attr, '')
                if subgraph_name in subgraphs:
                    graph = subgraphs[subgraph_name]
                else:
                    graph = vg
                self.add_op_to_graph(op, graph)

            for sg in subgraphs.values():
                vg.subgraph(sg)

        else:
            for op in visited_ops:
                self.add_op_to_graph(op, vg)

        tmp_dir = tempfile.mkdtemp()
        vg.render(directory=tmp_dir, view=True, cleanup=True)

        return ops, inits
