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

from collections import Iterable
from tensorflow.core.framework import graph_pb2
from ngraph.op_graph.serde.serde import _serialize_graph
from ngraph.op_graph.tensorboard.summary import _clean_tag


def ngraph_to_tf_graph_def(graph):
    # pylint: disable=line-too-long
    """
    Given an ngraph graph, convert it to a TensorFlow `GraphDef` protobuf object in memory.

    Arguments:
        graph (Op, Iterable): Ops to serialize as a TensorFlow Graph

    Returns:
        A Tensorflow `tensorflow.core.framework.graph_pb2.GraphDef` structure.

    References:
        Tensorflow graphdef proto:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto
        Tensorflow nodedef proto:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto
    """
    if not isinstance(graph, Iterable):
        graph = [graph]
    ng_graph_def = _serialize_graph(graph)
    op_names = set()
    op_uuid_map = dict()
    tf_graph_def = graph_pb2.GraphDef()
    for op in ng_graph_def.ops:
        node_def = tf_graph_def.node.add()
        node_def.name = _clean_tag(op.name)
        node_def.op = op.op_type
        if "_axes" in op.attrs:
            shape = node_def.attr["axes"].shape
            for axis in op.attrs["_axes"].axes.axes:
                dim = shape.dim.add()
                dim.name = str(axis.name)
                dim.size = int(axis.length)
        for key, value in op.attrs.items():
            if key.startswith("_ngraph_metadata_"):
                key = key.replace("_ngraph_metadata_", "")
                value_type = value.scalar.WhichOneof("value")
                if value_type == "string_val":
                    node_def.attr[key].s = str(value.scalar.string_val)
                elif value_type == "bool_val":
                    node_def.attr[key].b = value.scalar.bool_val
                elif value_type == "double_val":
                    node_def.attr[key].f = value.scalar.double_val
                elif value_type == "int_val":
                    node_def.attr[key].i = value.scalar.int_val
                else:
                    # TODO: Could also capture slice and dtype
                    pass

        if op.name in op_names:
            raise ValueError("Op with name {} exists in duplicate".format(op.name))
        op_names.add(op.name)
        op_uuid_map[op.uuid.uuid] = node_def

    for edge in ng_graph_def.edges:
        from_op = op_uuid_map[edge.from_uuid.uuid]
        to_op = op_uuid_map[edge.to_uuid.uuid]
        to_op.input.append(_clean_tag(from_op.name))

    return tf_graph_def
