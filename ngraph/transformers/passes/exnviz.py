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
from ngraph.op_graph.op_graph import TensorValueOp, AssignableTensorOp, IndexOp
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

    def arg_name(self, arg):
        return self.exop_name(arg.exop) + ':' + self.arg_ext(arg)

    def arg_ext(self, arg):
        return 'A' + str(arg.pos)

    def value_name(self, value):
        return self.exop_name(value.exop) + ':' + self.value_ext(value)

    def value_ext(self, value):
        return 'V' + str(value.pos)

    def tensor_name(self, tensor):
        if tensor not in self.tensors_with_nodes:
            self.tensors_without_nodes.add(tensor)
        return 'T' + str(id(tensor))

    def tensor_view_name(self, tensor_view):
        return self.tensor_name(tensor_view.tensor) + ':' + self.tensor_view_ext(tensor_view)

    def tensor_view_ext(self, tensor_view):
        return 'TV' + str(id(tensor_view))

    def add_edge(self, exop_from, exop_to, **kwargs):
        self.add_exop(exop_from)
        self.add_exop(exop_to)
        self.graph.edge(self.exop_name(exop_from),
                        self.exop_name(exop_to),
                        **kwargs)

    def add_control_edge(self, exop_from, exop_to, **kwargs):
        self.add_exop(exop_from)
        self.add_exop(exop_to)
        attrs = dict()
        attrs['weight'] = '10'
        self.graph.edge(self.exop_name(exop_from),
                        self.exop_name(exop_to),
                        color=control_edge_color,
                        **attrs)

    def add_arg_view_edge(self, arg):
        tensor_view = arg.read_view
        tensor = tensor_view.tensor
        self.add_tensor(tensor)
        # print('arg edge {} - {}'.format(self.tensor_view_name(tensor_view), self.arg_name(arg)))
        self.graph.edge(self.tensor_view_name(tensor_view),
                        self.arg_name(arg),
                        color=tensor_edge_color)

    def add_value_view_edge(self, value):
        tensor_view = value.write_view
        if tensor_view is None:
            return
        tensor = tensor_view.tensor
        self.add_tensor(tensor)
        # print(
        # 'val edge {} - {}'.format(self.value_name(value), self.tensor_view_name(tensor_view)))
        self.graph.edge(self.value_name(value),
                        self.tensor_view_name(tensor_view),
                        color=tensor_edge_color)

    def add_flow_edge(self, arg, **kwargs):
        value = arg.value
        self.add_exop(arg.exop)
        self.add_exop(value.exop)
        # print('flow edge {} - {}'.format(self.value_name(value), self.arg_name(arg)))
        self.graph.edge(self.value_name(value), self.arg_name(arg), **kwargs)
        if self.show_tensors:
            self.add_arg_view_edge(arg)

    def add_tensor(self, tensor):
        if tensor in self.tensors_with_nodes:
            return
        if tensor in self.tensors_without_nodes:
            self.tensors_without_nodes.remove(tensor)
        self.tensors_with_nodes.add(tensor)
        self.add_exop(tensor)

        views_labels = ' | '.join(['<{}>'.format(self.tensor_view_ext(tensor_view))
                                   for tensor_view in six.itervalues(tensor.tensor_descriptions)])
        label = '{ <tensor> ' + \
                tensor.tensor_description_base.name + \
                ' | { ' + views_labels + ' } }'
        self.graph.node(self.tensor_name(tensor), label=label, shape='Mrecord',
                        fillcolor=tensor_color, style='filled')
        if False:
            self.graph.edge(self.tensor_name(tensor), self.exop_name(tensor),
                            color=tensor_edge_color, style='dashed')

    def add_arg_tensor_view(self, arg):
        self.add_tensor(self, arg.read_view.tensor)

    def add_exop(self, exop, **kwargs):
        if exop in self.exops_with_nodes:
            return
        if exop in self.exops_without_nodes:
            self.exops_without_nodes.remove(exop)
        self.exops_with_nodes.add(exop)
        op = exop.op

        attrs = dict()
        # print('op {}'.format(type(op)))
        if isinstance(op, AssignableTensorOp):
            attrs['peripheries'] = '2'
            attrs['style'] = ''
        elif isinstance(op, IndexOp):
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

        if op is None:
            op_label = ""
        else:
            op_type_name = type(op).__name__
            if op_type_name in op.name:
                op_label = op.name
            else:
                op_label = '{}: {}'.format(op_type_name, op.name)

        if hasattr(op, 'axes') and self.show_axes:
            op_label += "\\n{}".format(str(op.axes))

        arg_label = ' | '.join(['<{}> {}'
                               .format(self.arg_ext(arg),
                                       arg.value.tensor.tensor_description_base.name)
                                for arg in exop.args])
        val_label = ' | '.join(['<{}> {}'
                               .format(self.value_ext(value),
                                       value.tensor.tensor_description_base.name)
                                for value in exop.values])
        label = '{ { ' + arg_label + ' } | <exop> ' + op_label + ' | { ' + val_label + ' } }'
        # print('{} label={}'.format(self.exop_name(exop), label))

        if self.show_tensors:
            for value in exop.values:
                self.add_value_view_edge(value)

        self.graph.node(self.exop_name(exop), label=label, shape='Mrecord', **attrs)

    def visit_exop(self, exop, *args):
        if not exop.next_exop.is_exop_end_of_list:
            self.add_control_edge(exop, exop.next_exop)
        for arg in exop.args:
            self.add_flow_edge(arg)
        if isinstance(exop.op, TensorValueOp):
            tv_exop = self.computation_decl.get_tensor(op=exop.op.value_tensor)
            self.add_edge(tv_exop, exop, color='blue')
            self.add_exop(tv_exop)

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
