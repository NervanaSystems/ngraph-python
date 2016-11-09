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
import ngraph as ng
from ngraph.op_graph.op_graph import AssignableTensorOp
from ngraph.op_graph.passes import PeepholeGraphPass, GraphPass, SimplePrune
from ngraph.util.generics import generic_method


def get_simple_graph():
    base_op = ng.constant(5.0)
    simple_graph = ng.log(ng.exp(base_op))
    return base_op, simple_graph


def test_simpleprune_graph_pass():
    base_op, simple_graph = get_simple_graph()
    output_graph = SimplePrune().do_pass([simple_graph])
    assert output_graph.pop() is base_op


class MySimplePeepholeGraphPass(PeepholeGraphPass):
    def __init__(self):
        super(MySimplePeepholeGraphPass, self).__init__()

    @generic_method
    def visit(self, op):
        pass

    @visit.on_type(AssignableTensorOp)
    def visit(self, op):
        if op.const == 5.0:
            self.replace_op(op, ng.constant(10.0))


class MySimpleGraphPass(GraphPass):
    def do_pass(self, ops):
        return len(ng.Op.ordered_ops(ops))


def test_simple_peephole():
    base_op, simple_graph = get_simple_graph()
    pass_inst = MySimplePeepholeGraphPass()
    output_graph = pass_inst.do_pass([simple_graph]).pop()
    assert output_graph.args[0].args[0].const == 10.0
