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
import logging

import six

from ngraph.transformers.passes.passes import GraphPass

from ngraph.op_graph.op_graph import Op
from ngraph.op_graph.serde import serde as ser
from ngraph.op_graph.serde import ops_pb2 as ops_pb


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
            for arg in op.all_deps:
                if arg not in visited:
                    frontier.add(arg)
                edges.append({'from': get_id(op), 'to': get_id(arg), 'color': 'blue'})

        with tempfile.TemporaryFile('w') as fid:
            json.dump(dict(nodes=nodes.values(), edges=edges), fid)


class VizPass(GraphPass):
    """
    A graph pass that visualizes nervana graphs and displays them to the user

    Parameters:
        subgraph_attr <string or None, default None>: A metadata attribute (eg: 'layer_type')
            that you wish to group ops by.
        show_axes <bool default False>: Whether to render axes information on nodes.
        show_all_metadata <bool, default False>: Whether to render all Op metadata on the nodes.
        view <bool, default True>: Whether to open the rendered PDF, if False, prints PDF location
            to stdout.

    """
    def __init__(self, subgraph_attr=None, show_axes=False, show_all_metadata=False, view=True):
        super(VizPass, self).__init__()
        self.show_axes = show_axes
        self.show_all_metadata = show_all_metadata
        self.subgraph_attr = subgraph_attr
        self.uuid_lookup_table = dict()
        self.view = view

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
        # Register op in lookup table by uuid for later edge creation
        self.uuid_lookup_table[op.uuid.get_bytes()] = op
        op_label = op.name
        if hasattr(op, 'axes') and self.show_axes:
            op_label += "\n{}".format(op.axes)
        if self.show_all_metadata:
            for k, v in six.iteritems(op.metadata):
                op_label += "\n{}={}".format(k, v)
        graph.node(op.name, op_label)

    def add_edge_to_graph(self, edge, graph):
        head_op = self.uuid_lookup_table[edge.from_uuid.uuid]
        tail_op = self.uuid_lookup_table[edge.to_uuid.uuid]
        if edge.edge_type == ops_pb.Edge.DATA:
            graph.edge(head_op.name, tail_op.name)
        elif edge.edge_type == ops_pb.Edge.CONTROL:
            graph.edge(head_op.name, tail_op.name, color='blue')
        elif edge.edge_type == ops_pb.Edge.CONTAINER:
            if '_ngraph_forward' in edge.attrs and head_op is not tail_op:
                graph.edge(head_op.name, tail_op.name, color='red', label='forward')
            else:  # ops
                graph.edge(head_op.name, tail_op.name, label='_ops', color='red', style='dotted')
        else:
            if '_ngraph_attribute' in edge.attrs:
                label = edge.attrs['_ngraph_attribute'].scalar.string_val
            else:
                label = edge.attrs['_ngraph_list_attribute'].scalar.string_val
            graph.edge(head_op.name, tail_op.name, label=label, color='red', style='dotted')

    def do_pass(self, ops, inits):
        try:
            import graphviz
        except ImportError:
            raise ImportError("You tried to use the ShowGraph transformer pass but did "
                              "not have the python graphviz library installed")
        # Get all ops and edges from this set
        all_ops = Op.all_op_references(ops)
        all_edges = ser._serialize_graph(ops).edges

        vg = graphviz.Digraph(node_attr={'shape': 'box'},
                              graph_attr={'nodesep': '.5',
                                          'ranksep': '.5'})
        if self.subgraph_attr is not None:
            subgraphs = {}
            for subgraph_name in self.get_subgraphs(all_ops):
                if subgraph_name not in subgraphs and subgraph_name is not None:
                    sg = graphviz.Digraph(name='cluster_{}'.format(subgraph_name))
                    sg.body.append('color="{}"'.format(self.random_color()))
                    sg.body.append('style=filled')
                    sg.body.append('label="{}"'.format(subgraph_name))
                    subgraphs[subgraph_name] = sg
            for op in all_ops:
                subgraph_name = op.metadata.get(self.subgraph_attr, '')
                if subgraph_name in subgraphs:
                    graph = subgraphs[subgraph_name]
                else:
                    graph = vg
                self.add_op_to_graph(op, graph)

            for sg in subgraphs.values():
                vg.subgraph(sg)

        else:
            for op in all_ops:
                self.add_op_to_graph(op, vg)

        for edge in all_edges:
            self.add_edge_to_graph(edge, vg)

        tmp_dir = tempfile.mkdtemp()
        vg.render(directory=tmp_dir, view=self.view, cleanup=True)
        if not self.view:
            logging.info("VizPass graph rendered to {}", tmp_dir)
        # Cleanup
        self.uuid_lookup_table.clear()

        return ops, inits
