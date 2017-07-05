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


def move_op(comp_exop, op_to_move):
    previous_op = None
    for exop in comp_exop:
        for value in op_to_move.values:
            if value.write_view.tensor in [arg.read_view.tensor for arg in exop.args]:
                # first op that needs new_op's output
                if op_to_move is not previous_op:
                    comp_exop.move_exop_to_after_exop(op_to_move, previous_op)
                return
        previous_op = exop


def is_parent(op, parent):
    parent_values = [x.tensor_decl for x in parent.values]
    for arg in op.args:
        if arg.value.tensor_decl in parent_values:
            return True
    return False


def is_child(op, child):
    child_args = [x.value.tensor_decl for x in child.args]
    for value in op.values:
        if value.tensor_decl in child_args:
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
        for op in exop_block:
            persistent = True
            for arg in op.args:
                if not arg.tensor.is_persistent:
                    persistent = False
            if persistent:
                if isinstance(op.op, ReadOp):
                    pass
                elif isinstance(op.op, WriteOp):
                    pass
                else:
                    persistent_ops.append(op)

        for op_to_move in persistent_ops:
            move_op(exop_block, op_to_move)
