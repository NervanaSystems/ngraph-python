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


def _remove_name_prefix(name):
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

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets importer states.
        """
        self.name_op_map = dict()
        self.ops_bridge = OpsBridge()
        self.init_ops = []
        self.graph_def = None

    def parse_protobuf(self, pb_file, verbose=False):
        """
        Parse graph_def protobuf file.

        Args:
            pb_file: protobuf file path
        """
        # read graph_def
        graph_def = tf.GraphDef()
        if mimetypes.guess_type(pb_file)[0] == 'text/plain':
            graph_def = tf.GraphDef()
            with open(pb_file, 'r') as f:
                text_format.Merge(f.read(), graph_def)
        else:
            with open(pb_file, 'rb') as f:
                graph_def.ParseFromString(f.read())

        self.parse_graph_def(graph_def, verbose=verbose)

    def parse_graph_def(self, graph_def, verbose=False):
        """
        Parse graph_def

        Args:
            graph_def: GraphDef object
        """
        self.graph_def = graph_def

        # pass 1: identify assigns connected to NoOps (hack around issue #373)
        # TODO: just a temp fix for now
        for tf_node in graph_def.node:
            if tf_node.op == 'NoOp' and tf_node.name == 'init':
                assign_op_names = set([_remove_name_prefix(name) for name in
                                       tf_node.input])
                self.ops_bridge.init_assign_op_names |= assign_op_names

        # pass 2: process nodes
        for tf_node in graph_def.node:
            # print node
            if verbose:
                print(tf_node)

            # resolve inputs
            input_ops = [self.get_op_handle_by_name(name) for name in tf_node.input]

            # get output op
            if None in input_ops:
                # ignored
                output_op = None
            else:
                # call bridge op
                output_op = self.ops_bridge(tf_node, input_ops)

            # convert to list for convenience
            if isinstance(output_op, tuple):
                output_op = list(output_op)
            else:
                output_op = [output_op]

            # post-process output ops
            for idx in range(len(output_op)):
                output_op[idx] = self.post_process_op(output_op[idx])

            # convert back to tuple or op
            if len(output_op) > 1:
                output_op = tuple(output_op)
            else:
                output_op = output_op[0]

            # save init node
            if tf_node.op == 'NoOp' and tf_node.name == 'init':
                self.init_ops.append(output_op)

            self.name_op_map[tf_node.name] = output_op

    def post_process_op(self, op):
        if op is None:
            return None
        # avoid illegal names in ngraph generated code
        op.name = op.name.replace("/", "_")

        # cast to new axes for safety: debugging purpose
        # import ngraph as ng
        # if hasattr(op, 'axes'):
        #     new_axes = [ng.make_axis(a.length) for a in op.axes]
        #     op = ng.cast_axes(op, axes=new_axes)
        return op

    def get_op_handle_by_name(self, name):
        """
        Get ngraph op from TF Node's name, supports multiple output node.

        Args:
            name: TF Node's name. For example, `BroadcastGradientArgs:1`
                  retrieves the index 1 output of the op
                  `gradients/mul_grad/BroadcastGradientArgs`

        Returns:
            Op: the corresponding ngraph op
        """
        # remove prefix
        name = _remove_name_prefix(name)

        # remove suffix of ":" for multiple output node
        name_splits = name.split(":")

        # get node
        if len(name_splits) > 1:
            # check
            assert len(name_splits) == 2
            # split
            idx = int(name_splits[1])
            name_truncated = name_splits[0]
            # get outputs
            outputs = self.name_op_map[name_truncated]
            # get by idx
            if not isinstance(outputs, tuple):
                assert idx == 0
                return self.name_op_map[name_truncated]
            else:
                return self.name_op_map[name_truncated][idx]
        else:
            return self.name_op_map[name]

    def get_op_handle(self, tf_op):
        """
        Get the matching tf op to ngraph op
        Args:
            tf_op: TensorFlow graph nodes

        Returns:
            Op: the corresponding ngraph op
        """
        if isinstance(tf_op, list):
            return [self.get_op_handle_by_name(op.name) for op in tf_op]
        else:
            return self.get_op_handle_by_name(tf_op.name)
