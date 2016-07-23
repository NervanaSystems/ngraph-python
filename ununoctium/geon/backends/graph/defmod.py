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
from geon.backends.graph.names import name_scope_list, with_name_scope
from geon.backends.graph.graph import Model, with_graph_scope, with_environment, \
    get_current_environment
from geon.backends.graph.environment import bound_environment

from geon.backends.graph.defmodimp import Axis, deriv, input, Variable, ArrayWithAxes
from geon.backends.graph.defmodimp import absolute, add, cos, divide, dot, exp, log, \
    maximum, minimum, multiply
from geon.backends.graph.defmodimp import negative, reciprocal, sig, sin, softmax, sqrt, \
    square, subtract, sum
from geon.backends.graph.defmodimp import tanh
from geon.backends.graph.defmodimp import doall, decrement

from geon.backends.graph.defmodimp import Axis, Tensor, Variable, RecursiveTensor, Var
from geon.backends.graph.defmodimp import get_all_defs, find_all
