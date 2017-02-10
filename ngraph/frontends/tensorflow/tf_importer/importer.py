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
from __future__ import print_function

# basics
import mimetypes
import sys
import os

# tensorflow
import tensorflow as tf
from google.protobuf import text_format

# importer
import ngraph as ng
from ngraph.frontends.tensorflow.tf_importer.ops_bridge import OpsBridge
from ngraph.frontends.tensorflow.tf_importer.utils import remove_tf_name_prefix


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
        # TF's graph and graph_def
        self._graph = None
        self._graph_def = None
        # name to op dict and obs bridge converter
        self._name_op_map = dict()
        self._ops_bridge = OpsBridge()
        # checkpoint path for weight import
        self._checkpoint_path = None

    def import_protobuf(self, pb_file, verbose=False):
        """
        Imports graph_def from protobuf file to ngraph.

        Arguments:
            pb_file: Protobuf file path.
            verbose: Prints graph_def at each node if True.
        """
        # read graph_def
        graph_def = tf.GraphDef()
        if mimetypes.guess_type(pb_file)[0] == 'text/plain':
            with open(pb_file, 'r') as f:
                text_format.Merge(f.read(), graph_def)
        else:
            with open(pb_file, 'rb') as f:
                graph_def.ParseFromString(f.read())

        self.import_graph_def(graph_def, verbose=verbose)

    def import_graph(self, graph, verbose=False):
        """
        Imports a graph to ngraph.

        Args:
            graph: TF's Graph object
            verbose: Prints graph_def at each node if True.
        """
        self._graph = graph
        self._graph_def = self._graph.as_graph_def()
        self.import_graph_def(self._graph_def, verbose=verbose)

    def import_graph_def(self, graph_def, verbose=False):
        """
        Imports a graph_def to ngraph.

        Arguments:
            graph_def: GraphDef object
            verbose: Prints graph_def at each node if True.
        """
        # process nodes
        for tf_node in graph_def.node:
            # print node
            # if verbose:
            # print(tf_node)

            # resolve inputs
            input_ops = [
                self.get_op_handle_by_name(name) for name in tf_node.input
            ]

            # get output op
            if None in input_ops:
                # ignored
                output_op = None
            else:
                # call bridge op
                output_op = self._ops_bridge(tf_node, input_ops)

            # convert to list for convenience
            if isinstance(output_op, tuple):
                output_op = list(output_op)
            else:
                output_op = [output_op]

            # post-process output ops
            for idx in range(len(output_op)):
                output_op[idx] = self._post_process_op(output_op[idx])

            # convert back to tuple or op
            if len(output_op) > 1:
                output_op = tuple(output_op)
            else:
                output_op = output_op[0]

            self._name_op_map[tf_node.name] = output_op

    def import_meta_graph(self, mata_graph_path, checkpoint_path=None):
        """
        Import metagrpah and checkpoint (optional)

        Arguments:
            mata_graph_path: path to MetaGraph file
            checkpoint_path: path to checkpoint file
        """
        self._checkpoint_path = checkpoint_path
        graph = tf.Graph()
        with graph.as_default():
            # restore from meta
            meta_graph_path = os.path.join(os.getcwd(), mata_graph_path)
            self.saver = tf.train.import_meta_graph(meta_graph_path)

            # parse graph
            self.import_graph(graph)

    def get_op_handle(self, tf_op):
        """
        Get the matching tf op to ngraph op

        Arguments:
            tf_op: TensorFlow graph node or a list of nodes.

        Returns:
            Op: the corresponding ngraph op or a list of ops.
        """
        if isinstance(tf_op, list):
            return [self.get_op_handle_by_name(op.name) for op in tf_op]
        else:
            return self.get_op_handle_by_name(tf_op.name)

    def get_op_handle_by_name(self, name):
        """
        Get ngraph op from TF Node's name, supports multiple output node.

        Arguments:
            name: TF Node's name. For example, `BroadcastGradientArguments1`
                  retrieves the index 1 output of the op
                  `gradients/mul_grad/BroadcastGradientArgs`

        Returns:
            Op: the corresponding ngraph op
        """
        # remove prefix
        name = remove_tf_name_prefix(name)

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
            outputs = self._name_op_map[name_truncated]
            # get by idx
            if not isinstance(outputs, tuple):
                assert idx == 0
                return self._name_op_map[name_truncated]
            else:
                return self._name_op_map[name_truncated][idx]
        else:
            return self._name_op_map[name]

    def get_collection_handle(self, names):
        """
        Get collection handles, collection are saved via `tf.add_to_collection`.

        Arguments:
            names: list of names

        Returns:
            List of ngraph ops corresponding to the names
        """
        if self._graph is None:
            raise ValueError("self.graph is empty, import meta_graph first.")
        with self._graph.as_default():
            tf_nodes = [tf.get_collection(name)[0] for name in names]
            ng_ops = [self.get_op_handle(tf_node) for tf_node in tf_nodes]
            return ng_ops

    def get_restore_op(self):
        """
        Get variable restoring ngraph op from TF model checkpoint

        Returns:
            A `ng.doall` op that restores the stored weights in TF model
            checkpoint
        """
        if self._graph is None:
            raise ValueError("self._graph is None, import meta_graph first.")
        if self._checkpoint_path is None:
            raise ValueError("self._checkpoint_path is None, please specify"
                             "checkpoint_path while importing meta_graph.")
        with self._graph.as_default():
            tf_variables = tf.global_variables()
            ng_variables = self.get_op_handle(tf_variables)
            ng_restore_ops = []
            with tf.Session() as sess:
                checkpoint_path = os.path.join(os.getcwd(), self._checkpoint_path)
                self.saver.restore(sess, checkpoint_path)
                for tf_variable, ng_variable in zip(tf_variables, ng_variables):
                    val = sess.run(tf_variable)
                    ng_restore_ops.append(ng.assign(ng_variable, val))
            return ng.doall(ng_restore_ops)

    def _post_process_op(self, op):
        """
        Replace op name for safety and cast op's axes if necessary.

        Arguments:
            op: A ngraph Op.

        Returns:
            Processed ngraph Op.
        """
        if op is None:
            return None
        # avoid illegal names in ngraph generated code
        op.name = op.name.replace("/", "_")

        # cast to new axes for safety: debugging purpose only
        # import ngraph as ng
        # if hasattr(op, 'axes'):
        #     new_axes = [ng.make_axis(a.length) for a in op.axes]
        #     op = ng.cast_axes(op, axes=new_axes)
        return op

    def _get_unimplemented_ops(self, pb_path):
        """
        Returns a list of unimplemented ops' names.

        Arguments:
            pb_path: Protobuf file path.

        Returns:
            List of unimplemented ops' names.
        """
        # get required op
        with open(pb_path) as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]

            required_ops = set()
            for line in lines:
                if line[:3] == 'op:':
                    op_name = line.split(' ')[1][1:-1]
                    required_ops.add(op_name)

        # get supported ops
        ob = OpsBridge()
        supported_ops = set([name for name in dir(ob)
                             if name[:1] != "_" and name not in ob.__dict__])

        # get unimplemented ops
        unimplemented_ops = required_ops - supported_ops
        return sorted(list(unimplemented_ops))


if __name__ == '__main__':
    # get unimplemented ops
    importer = TFImporter()
    ops = importer._get_unimplemented_ops(sys.argv[1])
    print(ops)
