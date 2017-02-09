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
from ngraph.op_graph.op_graph import as_op
from ngraph.transformers.passes.passes import SimplePrune
from ngraph.util.ordered import OrderedSet


def get_simple_graph():
    base_op = as_op(ng.constant(5.0))
    simple_graph = ng.log(ng.exp(base_op))
    return base_op, simple_graph


class StubTransformer(object):
    def __init__(self):
        self.state_initialization_ops = OrderedSet()

    def add_initialization_ops(self, ops):
        return False


def test_simpleprune_graph_pass():
    transformer = StubTransformer()
    base_op, simple_graph = get_simple_graph()
    SimplePrune().do_pass([simple_graph], transformer)
    assert simple_graph.forwarded is base_op
