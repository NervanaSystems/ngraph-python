# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
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

        # TODO Add other types when they are in use
        assert isinstance(self.exop_block, ExOpBlock)
        self.did_something = True
        while self.did_something:
            self.did_something = False
            for exop in self.exop_block:
                self.visit_exop(exop, *exop.input_decls)
        return

    @abc.abstractmethod
    def visit_exop(self, exop, *exop_args):
        pass

    def exop_args(self, exop):
        return (input_decl.source_output_decl.exop.op for input_decl in exop.input_decls)

    def op_args(self, op):
        return self.exop_args(self.computation_decl.get_exop(op))

    def replace_op(self, old_op, replacement_op):
        self.exop_block.replace_op(old_op, replacement_op)


class DeadCodeEliminationPass(SequentialExOpPass):
    @exop_method(dispatch_base_type=Op)
    def visit_exop(self, exop, *args):
        if exop.has_side_effects:
            return

        for op in self.computation_decl.computation_op.parameters:
            if self.computation_decl.get_exop(op) is exop:
                # Used as a parameter
                return

        is_dead = True
        for output_decl in exop.output_decls:
            if len(output_decl.user_input_decls) > 0:
                is_dead = False
                break

        if is_dead:
            # print("DEAD", exop.op)
            self.exop_block.remove_exop(exop)
            self.did_something = True


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
            current_exop = ExOp(computation_decl=self.computation_decl,
                                create_value=False,
                                op=ReadOp(
                                    axes=source_tensor.tensor_description_base.axes))
            current_exop.add_ref_op(exop.op)
            current_exop.add_output_decl(source_tensor)
            source_tensor.exop = exop
            self.exop_block.add_exop(current_exop, exop.prev_exop)
            self.tensor_map[source_tensor] = current_exop
        return current_exop

    @visit_exop.on_type(TensorValueOp)
    def visit_exop(self, exop):
        current_exop = self.current_exop(exop, self.execution_graph.get_tensor_decl(
            op=exop.op.value_tensor).source_tensor)
        current_output_decl = current_exop.output_decls[0]
        current_output_decl.tensor_decl.merge_flags(exop.output_decls[0].tensor_decl)
        for input_decl in set(exop.output_decls[0].user_input_decls):
            input_decl.source_output_decl = current_output_decl
        self.exop_block.remove_exop(exop)

    @visit_exop.on_type(AssignOp)
    def visit_exop(self, exop, tensor_input_decl, value_input_decl):
        source_tensor = tensor_input_decl.source_output_decl.tensor_decl.source_tensor
        current_exop = self.current_exop(exop, source_tensor)
        write_exop = ExOp(computation_decl=self.computation_decl,
                          op=WriteOp(axes=current_exop.op.axes))
        write_tensor_decl = write_exop.output_decls[0].tensor_decl
        write_tensor_decl.source_tensor = source_tensor
        write_exop.add_write_arg(write_exop.output_decls[0])
        write_exop.add_input_decl(current_exop.output_decls[0])
        write_exop.add_write_arg(write_exop.output_decls[0],
                                 tensor_input_decl.tensor_view_decl.tensor_description)
        write_exop.add_input_decl(value_input_decl.source_output_decl)
        self.exop_block.replace_exop(exop, write_exop)
        self.tensor_map[source_tensor] = write_exop

    @visit_exop.on_type(Fill)
    def visit_exop(self, exop, tensor_input_decl):
        source_tensor = tensor_input_decl.source_output_decl.tensor_decl.source_tensor
        current_exop = self.current_exop(exop, source_tensor)
        write_exop = ExOp(computation_decl=self.computation_decl,
                          op=WriteOp(axes=current_exop.op.axes))
        write_tensor_decl = write_exop.output_decls[0].tensor_decl
        write_tensor_decl.source_tensor = source_tensor
        write_exop.add_write_arg(write_exop.output_decls[0])
        write_exop.add_input_decl(current_exop.output_decls[0])
        scalar_op = literal_scalar_exop(scalar=exop.op.scalar,
                                        computation_decl=self.computation_decl)
        write_exop.add_write_arg(write_exop.output_decls[0],
                                 tensor_input_decl.tensor_view_decl.tensor_description)
        write_exop.add_input_decl(scalar_op.output_decls[0])
        self.exop_block.replace_exop(exop, write_exop)
        self.tensor_map[source_tensor] = write_exop

    def end_pass(self, **kwargs):
        super(SSAConversion, self).end_pass(**kwargs)
        for source_tensor_decl, current_exop in iteritems(self.tensor_map):
            if current_exop.output_decls[0].tensor_decl is source_tensor_decl:
                continue
            if not source_tensor_decl.is_output:
                continue
            copy_exop = ExOp(computation_decl=self.computation_decl,
                             create_value=False,
                             op=WriteOp(axes=[]))
            copy_exop.add_write_arg(source_tensor_decl.exop.output_decls[0])
            copy_exop.add_input_decl(current_exop.output_decls[0])
            self.exop_block.add_exop(copy_exop)


class IndexElision(SequentialExOpPass):
    @exop_method(dispatch_base_type=Op)
    def visit_exop(self, exop, *args):
        pass

    @visit_exop.on_type(IndexOp)
    def visit_exop(self, exop, input_decl):
        output_decl = exop.input_decls[0].source_output_decl
        exop.output_decls[0].tensor_description = exop.op.transform_tensor_description(
            output_decl.tensor_description)
        exop.output_decls[0].tensor_decl = output_decl.tensor_decl
        output_decl.exop.take_output_decl(exop.output_decls[0])
        self.exop_block.remove_exop(exop)


class CopyElimination(SequentialExOpPass):
    @exop_method(dispatch_base_type=Op)
    def visit_exop(self, exop, *args):
        pass

    @visit_exop.on_type(WriteOp)
    def visit_exop(self, exop, *args):
        if len(args) == 1:
            source_output_decl = exop.input_decls[0].source_output_decl
            source_exop = source_output_decl.exop
            if source_exop.write_args and source_exop.input_decls:
                # Check if the persistent tensor written to is live
                # at the point where the temporary write is created
                # and skip this copy
                nexop = source_exop.next_exop
                in_use = False
                while (nexop is not exop) and (not nexop.is_exop_end_of_list):
                    for input_decl in nexop.input_decls:
                        if exop.write_args[0].tensor_decl is input_decl.tensor_decl:
                            in_use = True
                    nexop = nexop.next_exop
                if in_use:
                    return

                update_tensor_description = source_exop.write_args[1].tensor_description
                del source_exop.write_args[:]
                if source_exop.input_decls[0].tensor_decl is exop.write_args[0].tensor_decl:
                    source_exop.add_write_arg(exop.write_args[0].source_output_decl,
                                              update_tensor_description)
                    source_exop.input_decls.pop(0)
                else:
                    source_exop.add_write_arg(exop.write_args[0].source_output_decl)
                    source_exop.add_write_arg(exop.write_args[0].source_output_decl,
                                              update_tensor_description)
                self.exop_block.replace_output_decl(source_exop.output_decls[0],
                                                    exop.write_args[0].source_output_decl)
                self.exop_block.remove_exop(exop)
