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
from cachetools.func import lru_cache

from ngraph.frontends.onnx.onnx_importer.ops_bridge import make_ng_nodes
from ngraph.frontends.tensorflow.tf_importer.utils_pos_axes import make_pos_axes


class WrapperBaseClass(object):
    """Base class for wrappers, which add ngraph import functionality to ONNX protobuf objects."""

    def __init__(self, onnx_proto_instance, graph):
        # type: (protobuf.message.Message, GraphWrapper) -> None
        self._proto = onnx_proto_instance
        self._graph = graph

    def __getattr__(self, name):  # type: (str) -> Any
        return getattr(self._proto, name)

    def __repr__(self):  # type: () -> str
        name = getattr(self._proto, 'name', None)
        if name:
            return '<{}: {}>'.format(self.__class__.__name__, name)
        return '<{}>'.format(self.__class__.__name__)


class ModelWrapper(WrapperBaseClass):
    """Wrapper for ONNX ModelProto objects."""

    def __init__(self, model_proto):  # type: (onnx.ModelProto) -> None
        self.graph = GraphWrapper(model_proto.graph)
        super(ModelWrapper, self).__init__(model_proto, self.graph)


class GraphWrapper(WrapperBaseClass):
    """
    Wrapper for ONNX GraphProto objects.

    Transforms objects defined in an ONNX graph to ngraph tensors and nodes.
    """

    def __init__(self, onnx_proto_instance):  # type: (onnx.GraphProto) -> None
        super(GraphWrapper, self).__init__(onnx_proto_instance, self)
        self._ng_node_cache = {}  # type: Dict[str, TensorOp]
        self.node = [NodeWrapper(node, self) for node in self._proto.node]
        self.input = [ValueInfoWrapper(inpt, self) for inpt in self._proto.input]
        self.output = [ValueInfoWrapper(output, self) for output in self._proto.output]
        self.initializer = [TensorWrapper(initializer, self)
                            for initializer in self._proto.initializer]
        self._initialize_ng_tensors()
        self._initialize_ng_nodes()

    @lru_cache(maxsize=512)
    def get_initializer(self, value_name):  # type: (str) -> Optional[TensorWrapper]
        """
        Get a wrapped initializer tensor (if exists) for an input value.

        :param value_name: name of value string (ONNX namespace Value)
        :return: initializer tensor for value
        """
        return next((initializer for initializer in self.initializer
                     if initializer.name == value_name), None)

    @lru_cache(maxsize=512)
    def get_input(self, value_name):  # type: (str) -> Optional[ValueInfoWrapper]
        """
        Get a wrapped graph input value.

        :param value_name: string (ONNX namespace Value)
        :return: input value if found
        """
        return next((inpt for inpt in self.input if inpt.name == value_name), None)

    def ng_node_cache_get(self, name):  # type: (str) -> Optional[TensorOp]
        """
        Get an ngraph Op node from graph's cache.

        :param name: name of value string (ONNX namespace Value)
        :return: ngraph Op node if found
        """
        return self._ng_node_cache.get(name)

    def ng_node_cache_set(self, name, node):  # type: (str, TensorOp) -> None
        """
        Store an ngraph Op node in this graph's cache.

        :param name: name of value string (ONNX namespace Value)
        :param node: ngraph Op node
        """
        self._ng_node_cache[name] = node

    def _initialize_ng_tensors(self):  # type: () -> None
        """Create and cache ngraph Op nodes for all input values in the ONNX graph."""
        for value_info in self.input:
            value_info.get_ng_node()

    def _initialize_ng_nodes(self):  # type: () -> None
        """Create and cache ngraph Op nodes for all operation nodes in the ONNX graph."""
        for node in self.node:
            node.get_ng_nodes_dict()

    def get_input_names(self):  # type: () -> List[str]
        """Return a list of names of graph inputs.

        Inputs with an initializer will not be returned.
        """
        return [input_node.name for input_node in self._proto.input
                if not self.get_initializer(input_node.name)]

    def get_output_names(self):  # type: () -> List[str]
        """Return a list of names of graph outputs.

        If graph does not define outputs, return outputs of all graph nodes.
        """
        output_names = [output_node.name for output_node in self._proto.output]
        if not output_names:
            output_names = [node_name for node in self._proto.node for node_name in node.output]
        return output_names

    def get_ng_model(self):  # type: () -> List[Dict]
        """
        Get an ngraph output Op and input Ops for each output value of the ONNX graph.

        :return: a list of dicts with the imported ngraph model, e.g.:
            [{
                'name': 'Y',
                'inputs': [<AssignableTensorOp(placeholder):4552991464>],
                'output': <Abs(Abs_0):4552894504>
            }]
        """
        output_nodes = []
        for output_name in self.get_output_names():
            output_nodes.append({
                'name': output_name,
                'output': self.ng_node_cache_get(output_name),
                'inputs': [self.ng_node_cache_get(input_) for input_ in self.get_input_names()],
            })
        return output_nodes


class ValueInfoWrapper(WrapperBaseClass):
    """
    Wrapper for ONNX ValueInfoProto objects.

    Transforms values defined in an ONNX model to ngraph tensor ops.
    """

    @lru_cache(maxsize=1)
    def get_ng_axes(self):  # type: () -> Axes
        """Create an ngraph Axes object matching the shape of this value."""
        shape = []
        for dim in self.type.tensor_type.shape.dim:
            if dim.dim_param:
                raise NotImplementedError('Symbolic variable representation of '
                                          'tensor shape (dim_param) not supported yet.')
            shape.append(dim.dim_value)

        return ng.make_axes(axes=make_pos_axes(shape))

    def get_dtype(self):  # type: () -> numpy.dtype
        """Return the Numpy data type for this value."""
        return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[self.type.tensor_type.elem_type]

    @lru_cache(maxsize=1)
    def get_ng_placeholder(self):  # type: () -> TensorOp
        """Create an ngraph placeholder node for this value."""
        axes = self.get_ng_axes()
        dtype = self.get_dtype()
        return ng.placeholder(axes=axes, dtype=dtype).named(self.name)

    @lru_cache(maxsize=1)
    def get_ng_variable(self):  # type: () -> TensorOp
        """Create an ngraph variable node for this value."""
        axes = self.get_ng_axes()
        dtype = self.get_dtype()
        if self.has_initializer:
            initializer = self.get_initializer()
            return ng.variable(axes=axes, dtype=dtype,
                               initial_value=initializer.to_array()).named(self.name)
        return ng.variable(axes=axes, dtype=dtype).named(self.name)

    @lru_cache(maxsize=1)
    def get_ng_constant(self):  # type: () -> TensorOp
        """Create an ngraph variable node for this value."""
        axes = self.get_ng_axes()
        dtype = self.get_dtype()
        if not self.has_initializer:
            raise ValueError('Cannot create a constant without an initial value.')

        return ng.constant(self.get_initializer().to_array(), axes=axes, dtype=dtype)

    def get_ng_node(self):  # type: () -> TensorOp
        """Create an ngraph placeholder or variable node for this value."""
        cached_node = self._graph.ng_node_cache_get(self.name)
        if cached_node:
            return cached_node

        if self.has_initializer:
            node = self.get_ng_constant()
        else:
            node = self.get_ng_placeholder()

        self._graph.ng_node_cache_set(self.name, node)
        return node

    @property
    def has_initializer(self):  # type: () -> bool
        """Return true if the ONNX graph contains an initializer corresponding to this value."""
        if self.get_initializer():
            return True
        return False

    def get_initializer(self):  # type: () -> TensorWrapper
        """Get the initializer tensor corresponding to this value."""
        return self._graph.get_initializer(self.name)


class TensorWrapper(WrapperBaseClass):
    """Wrapper for ONNX TensorProto objects."""

    def to_array(self):  # type: () -> numpy.ndarray
        """Get the value of this tensor as a Numpy array."""
        return onnx.numpy_helper.to_array(self._proto)


class NodeWrapper(WrapperBaseClass):
    """Wrapper for ONNX NodeProto objects."""

    def __init__(self, onnx_proto_instance, graph):  # type: (onnx.NodeProto, GraphWrapper) -> None
        super(NodeWrapper, self).__init__(onnx_proto_instance, graph)
        self.input = [self._graph.get_input(input_name) for input_name in self._proto.input]
        self.output = [self._graph.get_input(output_name) for output_name in self._proto.output]
        self.attribute = [AttributeWrapper(attr, self._graph) for attr in self._proto.attribute]

    def __repr__(self):  # type: () -> str
        name = getattr(self._proto, 'name', None)
        op_type = self._proto.op_type
        if name:
            return '<{}({}): {}>'.format(self.__class__.__name__, op_type, name)
        return '<{}({})>'.format(self.__class__.__name__, op_type)

    def get_attribute(self, attribute_name):  # type: (str) -> Optional[AttributeWrapper]
        """
        Get a wrapped attribute of this node, if it exists.

        :param attribute_name: string (ONNX namespace Attribute)
        """
        return next((attr for attr in self.attribute if attr.name == attribute_name), None)

    def get_attribute_value(self, attribute_name, default=None):  # type: (str, Any) -> Any
        """
        Get the value of an attribute of this node.

        :param attribute_name: string (ONNX namespace Attribute)
        :param default: the default value to return if the attribute does not exist
        """
        attribute = self.get_attribute(attribute_name)
        if attribute:
            return attribute.get_value()
        return default

    def get_output_names(self):  # type: () -> List[str]
        """Get names of all outputs of this node."""
        return list(self._proto.output)

    def get_ng_inputs(self):  # type: () -> List[TensorOp]
        """Get a list of ngraph Ops for each input of this node."""
        return [self._graph.ng_node_cache_get(input_name) for input_name in self._proto.input]

    def _get_ng_nodes_dict_from_cache(self):  # type: () -> Dict[str, TensorOp]
        output_nodes_dict = {}
        for output_name in self._proto.output:
            ng_node = self._graph.ng_node_cache_get(output_name)
            if ng_node:
                output_nodes_dict.update({output_name: ng_node})
            else:
                return {}  # If any outputs are missing, we invalidate cache for this node
        return output_nodes_dict

    def get_ng_nodes_dict(self):  # type: () -> Dict[str, TensorOp]
        """
        Get a dict containing an ngraph Op for each output of this node.

        :return: dict {output_name: ng_node}
        """
        output_nodes_dict = self._get_ng_nodes_dict_from_cache()
        if output_nodes_dict:
            return output_nodes_dict

        output_node_names = self._proto.output
        output_nodes = make_ng_nodes(self)
        for output_name, node in zip(output_node_names, output_nodes):
            output_nodes_dict.update({output_name: node.named(output_name)})
            self._graph.ng_node_cache_set(output_name, node)

        return output_nodes_dict


class AttributeWrapper(WrapperBaseClass):
    """Wrapper for ONNX AttributeProto objects."""

    def get_value(self):  # type: () -> Any
        """Get the value of this attribute."""
        attr = self._proto
        if attr.HasField('f'):
            return attr.f
        elif attr.HasField('i'):
            return attr.i
        elif attr.HasField('s'):
            return attr.s.decode('utf-8')
        elif attr.HasField('t'):
            return TensorWrapper(attr.t, self._graph)
        elif attr.floats:
            return list(attr.floats)
        elif attr.ints:
            return list(attr.ints)
        elif attr.strings:
            return [string.decode('utf-8') for string in attr.strings]
        elif attr.tensors:
            return [TensorWrapper(t, self._graph) for t in attr.tensors]
        else:
            raise TypeError('Could not parse value for attribute %s', self.name)
