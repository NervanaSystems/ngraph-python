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
"""
Load TensorFlow's computation graph in protobuf and convert it to Neon's AST graph

python sample/create_sample_graph.py

If the protobuf (.pb) file does not exist, you might need to first run the TensorFlow script to
generate the protobuf file:
te
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import geon as be
from util.importer import create_nervana_graph
from geon.backends.graph.environment import Environment


def test_create_nervana_graph(pb_file, execute=False):
    env = Environment()
    graph = create_nervana_graph(pb_file, env)

    trans = be.NumPyTransformer()
    comp = trans.computation(graph.last_op)
    trans.finalize()
    trans.dataflow.view()

    if execute:
        with be.bound_environment(env):
            result = comp()
            print(result)

test_create_nervana_graph("sample/constant_graph.pb", True)
test_create_nervana_graph("sample/variable_graph_froze.pb", False)
test_create_nervana_graph("sample/variable_graph.pb", True)
