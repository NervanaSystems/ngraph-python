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

from ngraph.transformers.exop import ExOpBlock
from ngraph.op_graph.op_graph import WriteOp, ReadOp
from ngraph.transformers.passes.passes import GraphPass
from ngraph.op_graph.comm_nodes import CommunicationOp


def move_op(comp_exop, exop_to_move):
    previous_exop = None
    for exop in comp_exop:
        for output_decl in exop_to_move.output_decls:
            if output_decl.tensor_view_decl.tensor in [input_decl.read_view.tensor for input_decl
                                                       in exop.input_decls]:
                # first op that needs new_op's output
                if exop_to_move is not previous_exop:
                    comp_exop.move_exop_to_after_exop(exop_to_move, previous_exop)
                return
        previous_exop = exop


def is_parent(exop, parent_exop):
    parent_values = [output_decl.tensor_decl for output_decl in parent_exop.output_decls]
    for input_decl in exop.input_decls:
        if input_decl.source_output_decl.tensor_decl in parent_values:
            return True
    return False


def is_child(exop, child):
    child_args = [input_decl.source_output_decl.tensor_decl for input_decl in child.input_decls]
    for output_decl in exop.output_decls:
        if output_decl.tensor_decl in child_args:
            return True
    return False


class MemOptimizePass(GraphPass):
    def do_pass(self, computation_decl, **kwargs):
        self.computation_decl = computation_decl

        assert isinstance(computation_decl.exop_block, ExOpBlock)

        # self.optimize_persistent_input()
        self.gravity_shuffle()

    def move_op_up(self, op):
        prev = op.prev_exop
        while prev.is_exop_end_of_list is False:
            if is_parent(op, prev):
                if op != prev:
                    self.computation_decl.exop_block.move_exop_to_after_exop(op, prev)
                break
            prev = prev.prev_exop

    def move_op_down(self, op):
        next = op.next_exop
        while next.is_exop_end_of_list is False:
            if is_child(op, next):
                if op != next.prev_exop:
                    self.computation_decl.exop_block.move_exop_to_after_exop(op, next.prev_exop)
                break
            next = next.next_exop

    def gravity_shuffle(self):
        move_down = list()
        move_up = list()
        exop_block = self.computation_decl.exop_block
        for exop in exop_block:
            mass = 0
            if isinstance(exop.op, ReadOp):
                pass
            elif isinstance(exop.op, WriteOp):
                pass
            elif isinstance(exop.op, CommunicationOp):
                pass
            else:
                for tensor in exop.liveness_new_list:
                    if tensor.is_persistent is False:
                        mass += tensor.size
                for tensor in exop.liveness_free_list:
                    if tensor.is_persistent is False:
                        mass -= tensor.size
                if mass > 0:
                    move_down.append(exop)
                elif mass < 0:
                    move_up.append(exop)

        for op in move_down:
            self.move_op_down(op)

        for op in move_up:
            self.move_op_up(op)

    def optimize_persistent_input(self):
        exop_block = self.computation_decl.exop_block
        persistent_ops = list()
        for exop in exop_block:
            persistent = True
            for input_decl in exop.input_decls:
                if not input_decl.tensor.is_persistent:
                    persistent = False
            if persistent:
                if isinstance(exop.op, ReadOp):
                    pass
                elif isinstance(exop.op, WriteOp):
                    pass
                else:
                    persistent_ops.append(exop)

        for op_to_move in persistent_ops:
            move_op(exop_block, op_to_move)
