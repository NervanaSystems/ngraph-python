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
import abc

from future.utils import with_metaclass, iteritems
from ngraph.transformers.exop import ExOpBlock, ExOp, literal_scalar_exop
from ngraph.transformers.passes.passes import GraphPass

from ngraph.op_graph.op_graph import Op, TensorValueOp, AssignOp, IndexOp, Fill, \
    ReadOp, WriteOp
from ngraph.util.generics import TypeMethods


def exop_method(dispatch_base_type=object, extends=None, next_method_arg=None):
    """
    Makes a method generic on its exop.op argument.

    Returns:
        The generic method
    """
    return TypeMethods(extends=extends,
                       dispatch_type_fun=lambda self, exop, *args, **kwargs: type(exop.op),
                       dispatch_base_type=dispatch_base_type,
                       next_method_arg=next_method_arg)


class SequentialExOpPass(with_metaclass(abc.ABCMeta, GraphPass)):
    def __init__(self, **kwargs):
        super(SequentialExOpPass, self).__init__(**kwargs)
        self.did_something = False
        self.computation_decl = None
        self.execution_graph = None

    def do_pass(self, computation_decl, **kwargs):
        self.computation_decl = computation_decl
        self.execution_graph = computation_decl.execution_graph
        # TODO when more than one block, we would iterate over each block
        self.exop_block = computation_decl.exop_block

        self.begin_pass(**kwargs)

        # TODO Add other types when they are in use
        assert isinstance(self.exop_block, ExOpBlock)
        self.did_something = True
        while self.did_something:
            self.did_something = False
            for exop in self.exop_block:
                self.visit_exop(exop, *exop.args)

        return self.end_pass(**kwargs)

    @abc.abstractmethod
    def visit_exop(self, exop, *exop_args):
        pass

    def exop_args(self, exop):
        return (arg.value.exop.op for arg in exop.args)

    def op_args(self, op):
        return self.exop_args(self.computation_decl.get_exop(op))

    def replace_op(self, old_op, replacement_op):
        self.exop_block.replace_op(old_op, replacement_op)


class SSAConversion(SequentialExOpPass):
    def __init__(self, **kwargs):
        super(SSAConversion, self).__init__(**kwargs)

    def begin_pass(self, **kwargs):
        super(SSAConversion, self).begin_pass(**kwargs)
        self.tensor_map = dict()

    @exop_method(dispatch_base_type=Op)
    def visit_exop(self, exop, *args):
        pass

    def current_exop(self, exop, source_tensor):
        current_exop = self.tensor_map.get(source_tensor, None)
        if current_exop is None:
            current_exop = ExOp(computation_graph=self.computation_decl,
                                create_value=False,
                                op=ReadOp(
                                    axes=source_tensor.tensor_description_base.axes))
            current_exop.add_ref_op(exop.op)
            current_exop.add_value(source_tensor)
            source_tensor.exop = exop
            self.exop_block.add_exop(current_exop, exop.prev_exop)
            self.tensor_map[source_tensor] = current_exop
        return current_exop

    @visit_exop.on_type(TensorValueOp)
    def visit_exop(self, exop):
        current_exop = self.current_exop(exop, self.execution_graph.get_tensor_decl(
            op=exop.op.value_tensor).source_tensor)
        current_value = current_exop.values[0]
        current_value.tensor_decl.merge_flags(exop.values[0].tensor_decl)
        for arg in set(exop.values[0].value_users):
            arg.value = current_value
        self.exop_block.remove_exop(exop)

    @visit_exop.on_type(AssignOp)
    def visit_exop(self, exop, tensor_arg, value_arg):
        source_tensor = tensor_arg.value.tensor_decl.source_tensor
        current_exop = self.current_exop(exop, source_tensor)
        write_exop = ExOp(computation_graph=self.computation_decl,
                          op=WriteOp(axes=current_exop.op.axes))
        write_tensor = write_exop.values[0].tensor_decl
        write_tensor.source_tensor = source_tensor
        write_exop.add_write_arg(write_exop.values[0])
        write_exop.add_arg(current_exop.values[0])
        write_exop.add_write_arg(write_exop.values[0], tensor_arg.read_view.tensor_description)
        write_exop.add_arg(value_arg.value)
        self.exop_block.replace_exop(exop, write_exop)
        self.tensor_map[source_tensor] = write_exop

    @visit_exop.on_type(Fill)
    def visit_exop(self, exop, tensor_arg):
        source_tensor = tensor_arg.value.tensor_decl.source_tensor
        current_exop = self.current_exop(exop, source_tensor)
        write_exop = ExOp(computation_graph=self.computation_decl,
                          op=WriteOp(axes=current_exop.op.axes))
        write_tensor = write_exop.values[0].tensor_decl
        write_tensor.source_tensor = source_tensor
        write_exop.add_write_arg(write_exop.values[0])
        write_exop.add_arg(current_exop.values[0])
        scalar_op = literal_scalar_exop(scalar=exop.op.scalar,
                                        computation_graph=self.computation_decl)
        write_exop.add_write_arg(write_exop.values[0], tensor_arg.read_view.tensor_description)
        write_exop.add_arg(scalar_op.values[0])
        self.exop_block.replace_exop(exop, write_exop)
        self.tensor_map[source_tensor] = write_exop

    def end_pass(self, **kwargs):
        super(SSAConversion, self).end_pass(**kwargs)
        for source_tensor_decl, current_exop in iteritems(self.tensor_map):
            if current_exop.values[0].tensor_decl is source_tensor_decl:
                continue
            if not source_tensor_decl.is_output:
                continue
            copy_exop = ExOp(computation_graph=self.computation_decl,
                             create_value=False,
                             op=WriteOp(axes=[]))
            copy_exop.add_write_arg(source_tensor_decl.exop.values[0])
            copy_exop.add_arg(current_exop.values[0])
            self.exop_block.add_exop(copy_exop)


class IndexElision(SequentialExOpPass):
    @exop_method(dispatch_base_type=Op)
    def visit_exop(self, exop, *args):
        pass

    @visit_exop.on_type(IndexOp)
    def visit_exop(self, exop, arg):
        value = exop.args[0].value
        exop.values[0].tensor_description = exop.op.transform_tensor_description(
            value.tensor_description)
        exop.values[0].tensor_decl = value.tensor_decl
        value.exop.take_value(exop.values[0])
        self.exop_block.remove_exop(exop)
