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

import pytest
import onnx

import numpy as np
import ngraph as ng

from onnx.helper import make_node, make_graph, make_tensor_value_info, make_model, make_tensor
from ngraph.frontends.onnx.onnx_importer.model_wrappers import ModelWrapper, GraphWrapper, \
    NodeWrapper, ValueInfoWrapper, TensorWrapper


@pytest.fixture
def onnx_model():
    node = make_node('Add', ['X', 'Y'], ['Z'], name='test_node')
    graph = make_graph([node], 'test_graph',
                       [make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 2]),
                        make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 2])],
                       [make_tensor_value_info('Z', onnx.TensorProto.FLOAT, [1, 2])],
                       initializer=[make_tensor('X', onnx.TensorProto.FLOAT, [1, 2], [1, 1])])
    model = make_model(graph, producer_name='ngraph ONNXImporter')
    return model


def test_model_wrapper(onnx_model):
    wrapped_model = ModelWrapper(onnx_model)
    assert wrapped_model.producer_name == 'ngraph ONNXImporter'
    assert wrapped_model.graph.__class__ == GraphWrapper


def test_graph_wrapper(onnx_model):
    wrapped_model = ModelWrapper(onnx_model)
    wrapped_graph = wrapped_model.graph

    assert len(wrapped_graph.node) == 1
    assert wrapped_graph.node[0].__class__ == NodeWrapper

    assert len(wrapped_graph.input) == 2
    assert wrapped_graph.input[0].__class__ == ValueInfoWrapper

    assert len(wrapped_graph.output) == 1
    assert wrapped_graph.output[0].__class__ == ValueInfoWrapper

    assert len(wrapped_graph.initializer) == 1
    assert wrapped_graph.initializer[0].__class__ == TensorWrapper

    initializer = wrapped_graph.get_initializer('X')
    assert np.all(initializer.to_array() == np.array([[1., 1.]], dtype=np.float32))
    assert not wrapped_graph.get_initializer('Y')

    ng_model = wrapped_graph.get_ng_model()[0]
    assert ng_model['output'].__class__ == ng.op_graph.op_graph.Add
    assert ng_model['inputs'][0].__class__ == ng.op_graph.op_graph.AssignableTensorOp


def test_value_info_wrapper(onnx_model):
    wrapped_model = ModelWrapper(onnx_model)
    wrapped_value_info = wrapped_model.graph.input[0]

    assert wrapped_value_info.get_dtype() == np.float32
    assert wrapped_value_info.has_initializer

    initializer = wrapped_value_info.get_initializer()
    assert np.all(initializer.to_array() == np.array([[1., 1.]], dtype=np.float32))

    axes = wrapped_value_info.get_ng_axes()
    assert len(axes) == 2
    assert axes[1].length == 2

    placeholder = wrapped_value_info.get_ng_placeholder()
    assert placeholder.__class__ == ng.op_graph.op_graph.AssignableTensorOp
    assert placeholder.is_placeholder
    assert placeholder.axes == axes

    variable = wrapped_value_info.get_ng_variable()
    assert variable.__class__ == ng.op_graph.op_graph.AssignableTensorOp
    assert variable.is_trainable
    assert variable.axes == axes

    constant = wrapped_value_info.get_ng_constant()
    assert constant.__class__ == ng.op_graph.op_graph.AssignableTensorOp
    assert constant.is_constant
    assert constant.axes == axes

    ng_node = wrapped_value_info.get_ng_node()
    assert ng_node == constant


def test_node_wrapper(onnx_model):
    wrapped_model = ModelWrapper(onnx_model)
    wrapped_node = wrapped_model.graph.node[0]

    ng_inputs = wrapped_node.get_ng_inputs()
    assert len(ng_inputs) == 2
    assert ng_inputs[0].__class__ == ng.op_graph.op_graph.AssignableTensorOp

    ng_outputs = wrapped_node.get_ng_nodes_dict()
    assert len(ng_outputs) == 1
    assert ng_outputs['Z'].__class__ == ng.op_graph.op_graph.Add


def test_attribute_wrapper():
    def attribute_value_test(attribute_value):
        node = make_node('Abs', ['X'], [], name='test_node', test_attribute=attribute_value)
        model = make_model(make_graph([node], 'test_graph', [
            make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 2]),
        ], []), producer_name='ngraph')
        wrapped_attribute = ModelWrapper(model).graph.node[0].get_attribute('test_attribute')
        return wrapped_attribute.get_value()

    tensor = make_tensor('test_tensor', onnx.TensorProto.FLOAT, [1], [1])

    assert attribute_value_test(1) == 1
    assert type(attribute_value_test(1)) == np.long
    assert attribute_value_test(1.0) == 1.0
    assert type(attribute_value_test(1.0)) == np.float
    assert attribute_value_test('test') == 'test'
    assert attribute_value_test(tensor)._proto == tensor

    assert attribute_value_test([1, 2, 3]) == [1, 2, 3]
    assert attribute_value_test([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    assert attribute_value_test(['test1', 'test2']) == ['test1', 'test2']
    assert attribute_value_test([tensor, tensor])[1]._proto == tensor
