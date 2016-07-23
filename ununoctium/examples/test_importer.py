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
# ----------------------------------------------------------------------------
# load TensorFlow's computation graph in protobuf and convert it to Neon's AST graph
# ----------------------------------------------------------------------------
'''
Note that the TF package imported here is the local graphiti version.

If the protobuf (.pb) file does not exist, you might need to first run the TensorFlow script to
generate the protobuf file:

python ../../tf_benchmark/create_sample_graph.py

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import geon.backends.graph.funs as be
from util.importer import create_neon_graph
import geon.backends.graph.analysis as analysis
from geon.backends.graph.environment import Environment


def test_create_neon_graph(pb_file, execute=False):
    print("loading graph")
    graph_def = tf.GraphDef()

    env = Environment()

    with open(pb_file, 'rb') as f:
        graph_def.ParseFromString(f.read())  # read serialized binary file only

    ast_graph = create_neon_graph(graph_def, env)
    print(ast_graph)

    dataflow = analysis.DataFlowGraph([ast_graph.last_op])
    dataflow.view()

    if execute:
        with be.bound_environment(env):
            enp = be.NumPyTransformer(results=[ast_graph.last_op])
            result = enp.evaluate()
            print(result[ast_graph.last_op])

# test_create_neon_graph("../../tf_benchmark/sample/constant_graph.pb", False)
# test_create_neon_graph("../../tf_benchmark/sample/variable_graph.pb", False)
test_create_neon_graph(
    "../../tf_benchmark/sample/variable_graph_froze.pb", True)
