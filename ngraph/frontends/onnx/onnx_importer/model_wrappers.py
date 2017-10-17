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

from __future__ import print_function, division

import ngraph as ng
import onnx
import onnx.numpy_helper
import onnx.mapping
from onnx import onnx_pb2
from functools import lru_cache
from ngraph.frontends.onnx.onnx_importer.ops_bridge import OpsBridge
from ngraph.frontends.tensorflow.tf_importer.utils_pos_axes import make_pos_axes


class WrapperBaseClass:

    def __init__(self, onnx_proto_instance, graph):
        self._proto = onnx_proto_instance
        self._graph = graph

    def __getattr__(self, name):
        return getattr(self._proto, name)

    def __repr__(self):
        name = getattr(self._proto, 'name', None)
        if name:
            return "<{}: {}>".format(self.__class__.__name__, name)
        return "<{}>".format(self.__class__.__name__)


class ModelWrapper(WrapperBaseClass):

    def __init__(self, model_proto):
        self.graph = GraphWrapper(model_proto.graph)
        super(ModelWrapper, self).__init__(model_proto, self.graph)


class GraphWrapper(WrapperBaseClass):

    def __init__(self, onnx_proto_instance):  # type: (google.protobuf.message.Message) -> None
        super(GraphWrapper, self).__init__(onnx_proto_instance, self)
        self._ng_node_cache = {}
        self.node = [NodeWrapper(node, self) for node in self._proto.node]
        self.input = [ValueInfoWrapper(inpt, self) for inpt in self._proto.input]
        self.output = [ValueInfoWrapper(output, self) for output in self._proto.output]
        self.initializer = [TensorWrapper(initializer, self)
                            for initializer in self._proto.initializer]
        self.initialize_ng_tensors()
        self.initialize_ng_nodes()

    @lru_cache(maxsize=512)
    def get_initializer(self, value_name):
        return next((initializer for initializer in self.initializer
                     if initializer.name == value_name), None)

    @lru_cache(maxsize=512)
    def get_input(self, value_name):
        return next((inpt for inpt in self.input if inpt.name == value_name), None)

    def ng_node_cache_get(self, name):
        return self._ng_node_cache.get(name)

    def ng_node_cache_set(self, name, node):
        self._ng_node_cache[name] = node

    def initialize_ng_tensors(self):
        for value_info in self.input:
            value_info.get_ng_node()

    def initialize_ng_nodes(self):
        # @TODO: Verify topological sort of nodes
        for node in self.node:
            node.get_ng_nodes_dict()

    def get_ng_model(self):
        output_nodes = []
        for output_proto in self._proto.output:
            output_nodes.append({
                'name': output_proto.name,
                'output': self.ng_node_cache_get(output_proto.name),
                'inputs': [self.ng_node_cache_get(inpt.name) for inpt in self._proto.input]
            })
        return output_nodes


class ValueInfoWrapper(WrapperBaseClass):

    @lru_cache(maxsize=1)
    def get_ng_axes(self):
        if self.type.sparse_tensor_type.elem_type != onnx_pb2.TensorProto.UNDEFINED:
            raise NotImplementedError('Sparse tensors (SparseTensorTypeProto) not supported yet.')

        shape = []
        for dim in self.type.tensor_type.shape.dim:
            if dim.dim_param:
                raise NotImplementedError('Symbolic variable representation of '
                                          'tensor shape (dim_param) not supported yet.')
            shape.append(dim.dim_value)

        return ng.make_axes(axes=make_pos_axes(shape))

    def get_dtype(self):
        if self.type.sparse_tensor_type.elem_type != onnx_pb2.TensorProto.UNDEFINED:
            raise NotImplementedError('Sparse tensors (SparseTensorTypeProto) not supported yet.')

        return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[self.type.tensor_type.elem_type]

    @lru_cache(maxsize=1)
    def get_ng_placeholder(self):
        axes = self.get_ng_axes()
        dtype = self.get_dtype()
        return ng.placeholder(axes=axes, dtype=dtype).named(self.name)

    @lru_cache(maxsize=1)
    def get_ng_variable(self):
        axes = self.get_ng_axes()
        dtype = self.get_dtype()
        if self.has_initializer:
            initializer = self.get_initializer()
            return ng.variable(axes=axes, dtype=dtype,
                               initial_value=initializer.to_array()).named(self.name)
        return ng.variable(axes=axes, dtype=dtype).named(self.name)

    def get_ng_node(self):
        node = self._graph.ng_node_cache_get(self.name)
        if node:
            return node

        if self.has_initializer:
            node = self.get_ng_variable()
        else:
            node = self.get_ng_placeholder()

        self._graph.ng_node_cache_set(self.name, node)
        return node

    @property
    def has_initializer(self):
        if self.get_initializer():
            return True
        return False

    def get_initializer(self):
        return self._graph.get_initializer(self.name)


class TensorWrapper(WrapperBaseClass):

    def to_array(self):
        return onnx.numpy_helper.to_array(self._proto)


class NodeWrapper(WrapperBaseClass):

    def __init__(self, onnx_proto_instance, graph):
        super(NodeWrapper, self).__init__(onnx_proto_instance, graph)
        self.input = [self._graph.get_input(input_name) for input_name in self._proto.input]
        self.output = [self._graph.get_input(output_name) for output_name in self._proto.output]
        self.attribute = [AttributeWrapper(attr, self._graph) for attr in self._proto.attribute]

    def __repr__(self):
        name = getattr(self._proto, 'name', None)
        op_type = self._proto.op_type
        if name:
            return "<{}({}): {}>".format(self.__class__.__name__, op_type, name)
        return "<{}({})>".format(self.__class__.__name__, op_type)

    def get_attribute(self, attribute_name):
        return next((attr for attr in self.attribute if attr.name == attribute_name), None)

    def get_attribute_value(self, attribute_name, default=None):
        attribute = self.get_attribute(attribute_name)
        if attribute:
            return attribute.get_value()
        return default

    def get_ng_inputs(self):
        return [self._graph.ng_node_cache_get(input_name) for input_name in self._proto.input]

    def get_ng_nodes_dict(self):
        output_nodes_dict = {}
        ops_bridge = OpsBridge()

        for output_name in self._proto.output:
            ng_node = self._graph.ng_node_cache_get(output_name)
            if not ng_node:
                ng_node = ops_bridge.get_ng_node(self).named(output_name)
                self._graph.ng_node_cache_set(output_name, ng_node)

            output_nodes_dict.update({output_name: ng_node})
        return output_nodes_dict


class AttributeWrapper(WrapperBaseClass):

    def get_value(self):
        attr = self._proto
        if attr.HasField("f"):
            return attr.f
        elif attr.HasField("i"):
            return attr.i
        elif attr.HasField("s"):
            return attr.s.decode("utf-8")
        elif attr.HasField("t"):
            return TensorWrapper(attr.t, self._graph)
        elif attr.floats:
            return list(attr.floats)
        elif attr.ints:
            return list(attr.ints)
        elif attr.strings:
            return [string.decode("utf-8") for string in attr.strings]
        elif attr.tensors:
            return [TensorWrapper(t, self._graph) for t in attr.tensors]
        else:
            raise TypeError('Could not parse value for attribute %s', self.name)
