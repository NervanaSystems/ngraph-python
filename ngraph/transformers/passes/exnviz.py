# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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

import tempfile
import six

from ngraph.transformers.passes.expass import SequentialExOpPass
from ngraph.op_graph.op_graph import TensorValueOp, IndexOp
from ngraph.transformers.exop import ExOp

tensor_color = 'green'
tensor_edge_color = 'green'

reshape_color = 'lightblue'

control_color = 'pink'
control_edge_color = 'red'


class ExVizPass(SequentialExOpPass):
    """
    A graph pass that visualizes nervana graphs and displays them to the user
    """
    # def __init__(self, subgraph_attr=None, show_axes=True, show_all_metadata=False,
    #              view=True, cleanup=True, filename='Digraph', output_directory='.'):
    def __init__(self, **kwargs):
        super(ExVizPass, self).__init__()
        self.show_axes = kwargs.pop('show_axes', True)
        self.show_all_metadata = kwargs.pop('show_all_metadata', True)
        self.subgraph_attr = kwargs.pop('subgraph_attr', None)
        self.exops_with_nodes = set()
        self.exops_without_nodes = set()
        self.filename = kwargs.pop('filename', 'Digraph')
        self.view = kwargs.pop('view', True)
        self.cleanup = kwargs.pop('cleanup', True)
        self.show_tensors = kwargs.pop('show_tensors', False)
        output_directory = kwargs.pop('output_directory', '.')

        if self.view:
            if output_directory is None:
                output_directory = tempfile.mkdtemp()
        self.output_directory = output_directory

    def exop_name(self, exop):
        if exop not in self.exops_with_nodes:
            self.exops_without_nodes.add(exop)
        return 'E' + str(id(exop))

    def input_decl_name(self, input_decl):
        return self.exop_name(input_decl.exop) + ':' + self.input_decl_ext(input_decl)

    def input_decl_ext(self, input_decl):
        return 'A' + str(input_decl.pos)

    def output_decl_name(self, output_decl):
        return self.exop_name(output_decl.exop) + ':' + self.output_decl_ext(output_decl)

    def output_decl_ext(self, output_decl):
        return 'V' + str(output_decl.pos)

    def tensor_decl_name(self, tensor_decl):
        if tensor_decl not in self.tensors_with_nodes:
            self.tensors_without_nodes.add(tensor_decl)
        return 'T' + str(id(tensor_decl))

    def tensor_view_decl_name(self, tensor_view_decl):
        return self.tensor_decl_name(
            tensor_view_decl.tensor_decl) + ':' + self.tensor_view_decl_ext(
            tensor_view_decl)

    def tensor_view_decl_ext(self, tensor_view_decl):
        return 'TV' + str(id(tensor_view_decl))

    def add_tensor_decl_exop_edge(self, tensor_decl_from, exop_to, **kwargs):
        self.add_tensor_decl(tensor_decl_from)
        # self.add_node(exop_to)
        self.graph.edge(self.tensor_decl_name(tensor_decl_from),
                        self.exop_name(exop_to),
                        **kwargs)

    def add_control_edge(self, exop_from, exop_to, **kwargs):
        self.add_exop_node(exop_from)
        self.add_exop_node(exop_to)
        attrs = dict()
        attrs['weight'] = '10'
        self.graph.edge(self.exop_name(exop_from),
                        self.exop_name(exop_to),
                        color=control_edge_color,
                        **attrs)

    def add_input_decl_view_edge(self, input_decl):
        tensor_decl_view = input_decl.tensor_decl_view
        tensor_decl = tensor_decl_view.tensor_decl
        self.add_tensor_decl(tensor_decl)
        # print('arg edge {} - {}'.format(self.tensor_view_name(tensor_decl_view),
        #                                 self.input_decl_name(input_decl)))
        self.graph.edge(self.tensor_view_decl_name(tensor_decl_view),
                        self.input_decl_name(input_decl),
                        color=tensor_edge_color)

    def add_output_decl_view_edge(self, output_decl):
        tensor_view_decl = output_decl.tensor_view_decl
        if tensor_view_decl is None:
            return
        tensor = tensor_view_decl.tensor_decl
        self.add_tensor_decl(tensor)
        # print('val edge {} - {}'.format(self.output_decl_name(output_decl),
        #                                 self.tensor_decl_view_name(tensor_view_decl)))
        self.graph.edge(self.output_decl_name(output_decl),
                        self.tensor_view_decl_name(tensor_view_decl),
                        color=tensor_edge_color)

    def add_flow_edge(self, input_decl, **kwargs):
        output_decl = input_decl.source_output_decl
        self.add_exop_node(input_decl.exop)
        self.add_exop_node(output_decl.exop)
        # print('flow edge {} - {}'.format(self.output_decl_name(output_decl),
        #                                  self.input_decl_name(input_decl)))
        self.graph.edge(self.output_decl_name(output_decl),
                        self.input_decl_name(input_decl), **kwargs)
        if self.show_tensors:
            self.add_input_decl_view_edge(input_decl)

    def add_tensor_decl(self, tensor_decl):
        if tensor_decl in self.tensors_with_nodes:
            return
        if tensor_decl in self.tensors_without_nodes:
            self.tensors_without_nodes.remove(tensor_decl)
        self.tensors_with_nodes.add(tensor_decl)

        views_labels = ' | '.join(['<{}>'.format(self.tensor_view_decl_ext(tensor_view_decl))
                                   for tensor_view_decl in
                                   six.itervalues(tensor_decl.tensor_view_decls)])
        label = '{ <tensor> ' + tensor_decl.name + ' | { ' + views_labels + ' } }'
        self.graph.node(self.tensor_decl_name(tensor_decl), label=label, shape='Mrecord',
                        fillcolor=tensor_color, style='filled')
        if False:
            self.graph.edge(self.tensor_decl_name(tensor_decl), self.exop_name(tensor_decl),
                            color=tensor_edge_color, style='dashed')

    def add_exop_node(self, exop):
        if exop in self.exops_with_nodes:
            return
        if exop in self.exops_without_nodes:
            self.exops_without_nodes.remove(exop)
        self.exops_with_nodes.add(exop)

        attrs = dict()
        op = None

        op = exop.op
        # print('op {}'.format(type(op)))
        if isinstance(op, IndexOp):
            attrs['fillcolor'] = reshape_color
            attrs['style'] = 'filled'
        elif isinstance(op, TensorValueOp):
            attrs['fillcolor'] = reshape_color
            attrs['style'] = 'filled'
        elif isinstance(exop, ExOp):
            attrs['fillcolor'] = control_color
            attrs['style'] = 'filled'
        else:
            # attrs['fillcolor'] = control_color
            attrs['style'] = 'rounded'
        op_type_name = type(op).__name__
        if op_type_name in op.name:
            op_label = op.name
        else:
            op_label = '{}: {}'.format(op_type_name, op.name)
        arg_label = ' | '.join(['<{}> {}'
                               .format(self.input_decl_ext(input_decl),
                                       input_decl.source_output_decl.tensor_decl.name)
                                for input_decl in exop.input_decls])
        val_label = ' | '.join(['<{}> {}'
                               .format(self.output_decl_ext(output_decl),
                                       output_decl.tensor_decl.name)
                                for output_decl in exop.output_decls])
        label = '{ { ' + arg_label + ' } | <exop> ' + op_label + ' | { ' + val_label + ' } }'

        if hasattr(op, 'axes') and self.show_axes:
            op_label += "\\n{}".format(str(op.axes))

        # print('{} label={}'.format(self.exop_name(exop), label))

        if self.show_tensors:
            for output_decl in exop.values:
                self.add_output_decl_view_edge(output_decl)

        self.graph.node(self.exop_name(exop), label=label, shape='Mrecord', **attrs)

    def visit_exop(self, exop, *args):
        if not exop.next_exop.is_exop_end_of_list:
            self.add_control_edge(exop, exop.next_exop)
        for input_decl in exop.input_decls:
            self.add_flow_edge(input_decl)
        if isinstance(exop.op, TensorValueOp):
            tensor_decl = self.computation_decl.get_tensor_decl(op=exop.op.value_tensor)
            self.add_tensor_decl_exop_edge(tensor_decl, exop, color='blue')
            pass

    def begin_pass(self, filename=None, **kwargs):
        super(ExVizPass, self).begin_pass(**kwargs)
        try:
            import graphviz
        except ImportError:
            raise ImportError("You tried to use the ShowGraph transformer pass but did "
                              "not have the python graphviz library installed")
        if filename is None:
            filename = self.filename
        self.exops_with_nodes = set()
        self.exops_without_nodes = set()
        self.tensors_with_nodes = set()
        self.tensors_without_nodes = set()

        # Get all ops from this set
        self.graph = graphviz.Digraph(name=filename,
                                      # node_attr={'shape': 'box', 'style': 'rounded'},
                                      graph_attr={'nodesep': '.5',
                                                  'ranksep': '.5'})

    def end_pass(self, view=None, output_directory=None, cleanup=None, **kwargs):
        super(ExVizPass, self).end_pass(**kwargs)
        if view is None:
            view = self.view
        if output_directory is None:
            output_directory = self.output_directory
        if cleanup is None:
            cleanup = self.cleanup
        f = open(self.filename + '.dot', 'w')
        f.write(self.graph.source)
        f.close()
        self.graph.render(directory=output_directory, view=view, cleanup=cleanup)
