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
from geon.op_graph.names import name_scope_list, with_name_scope
from geon.frontends.base.graph import Model, with_graph_scope, with_environment, \
    get_current_environment
from geon.backends.graph.environment import bound_environment
from geon.op_graph.nodes import Node

from geon.frontends.declarative_graph.declarative_graph import Axis, deriv, input, Variable, ArrayWithAxes
from geon.frontends.declarative_graph.declarative_graph import absolute, add, cos, divide, dot, exp, log, \
    maximum, minimum, multiply
from geon.frontends.declarative_graph.declarative_graph import negative, reciprocal, sigmoid, sin, softmax, sqrt, \
    square, subtract, sum
from geon.frontends.declarative_graph.declarative_graph import tanh
from geon.frontends.declarative_graph.declarative_graph import doall, decrement

from geon.frontends.declarative_graph.declarative_graph import Axis, Tensor, Variable, RecursiveTensor, Var
from geon.frontends.declarative_graph.declarative_graph import get_all_defs, find_all
