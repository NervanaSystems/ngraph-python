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

# basics
from __future__ import absolute_import, division, print_function
import mimetypes

# tensorflow
import tensorflow as tf
from google.protobuf import text_format

# importer
from tf_importer.tf_importer.ops_bridge import OpsBridge

# ngraph
import ngraph as ng


def _strip_node_name(name):
    """
    Strip ^ from TF's node name
    Args:
        name: TF node name

    Returns:
        string: name with ^ stripped
    """
    return name[1:] if name[0] == "^" else name


class TFImporter:
    """
    Importer for Tensorflow GraphDef
    """

    def __init__(self, pb_file, verbose=False):
        # input fields
        self.pb_file = pb_file
        self.verbose = verbose

        # collections
        self.name_to_op = dict()  # maps TF node name to Neon op

        # bridge ops
        self.ops_bridge = OpsBridge()

        # init comp
        self.init_ops = []

        # read graph_def
        graph_def = tf.GraphDef()
        if mimetypes.guess_type(pb_file)[0] == 'text/plain':
            graph_def = tf.GraphDef()
            with open(pb_file, 'r') as f:
                text_format.Merge(f.read(), graph_def)
        else:
            with open(pb_file, 'rb') as f:
                graph_def.ParseFromString(f.read())

        # pass 1: identify assigns connected to NoOps (hack around issue #373)
        # TODO: just a temp fix for now
        for tf_node in graph_def.node:
            if tf_node.op == 'NoOp' and tf_node.name == 'init':
                assign_op_names = set([_strip_node_name(name) for name in
                                      tf_node.input])
                self.ops_bridge.init_assign_op_names |= assign_op_names

        # pass 2: process nodes
        for tf_node in graph_def.node:
            # print node
            if self.verbose:
                print(tf_node)

            # resolve inputs
            num_inputs = len(tf_node.input)
            input_ops = [self.name_to_op[_strip_node_name(tf_node.input[i])]
                         for i in range(num_inputs)]

            # call bridge op
            output_op = self.ops_bridge(tf_node, input_ops)

            # avoid illegal names in generated code
            output_op.name = output_op.name.replace("/", "_")

            # save to collections
            self.name_to_op[tf_node.name] = output_op

            # cast to new axes for safety
            if hasattr(output_op, 'axes'):
                new_axes = [ng.Axis(a.length) for a in output_op.axes]
                output_op = ng.AxesCastOp(output_op, axes=new_axes)

            # save init node
            if tf_node.op == 'NoOp' and tf_node.name == 'init':
                self.init_ops.append(output_op)
