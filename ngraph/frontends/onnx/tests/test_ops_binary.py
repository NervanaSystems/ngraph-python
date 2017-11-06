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
from cachetools.func import lru_cache

import onnx
import pytest

import ngraph as ng
import numpy as np
from onnx.helper import make_node, make_graph, make_tensor_value_info, make_model
from ngraph.frontends.onnx.onnx_importer.importer import import_onnx_model
from ngraph.frontends.tensorflow.tf_importer.utils_broadcast import broadcasted_shape


@lru_cache(maxsize=64)
def make_onnx_model_for_binary_op(op_type, input_shapes):
    output_shape = broadcasted_shape(*input_shapes)
    node = make_node(op_type, ["X", "Y"], ["Z"], name="test_node", broadcast=1)
    graph = make_graph([node], "test_graph",
                       [make_tensor_value_info("X", onnx.TensorProto.FLOAT, input_shapes[0]),
                        make_tensor_value_info("Y", onnx.TensorProto.FLOAT, input_shapes[1])],
                       [make_tensor_value_info("Z", onnx.TensorProto.FLOAT, output_shape)])
    model = make_model(graph, producer_name='ngraph ONNXImporter')
    return model


def import_and_compute(op_type, input_data_left, input_data_right):
    input_data_left = np.array(input_data_left)
    input_data_right = np.array(input_data_right)
    model = make_onnx_model_for_binary_op(op_type, (input_data_left.shape, input_data_right.shape))
    computation = import_model_make_computation(model)
    return computation(input_data_left, input_data_right)


@lru_cache(maxsize=1)
def get_transformer():
    return ng.transformers.make_transformer()


def import_model_make_computation(onnx_model):
    transformer = get_transformer()
    ng_model = import_onnx_model(onnx_model)[0]
    computation = transformer.computation(ng_model['output'], *ng_model['inputs'])
    return computation


def test_add():
    assert np.array_equal(import_and_compute('Add', 1, 2),
                          np.array(3, dtype=np.float32))

    assert np.array_equal(import_and_compute('Add', [1], [2]),
                          np.array([3], dtype=np.float32))

    assert np.array_equal(import_and_compute('Add', [1, 2], [3, 4]),
                          np.array([4, 6], dtype=np.float32))

    assert np.array_equal(import_and_compute('Add', [1, 2, 3], [4, 5, 6]),
                          np.array([5, 7, 9], dtype=np.float32))

    assert np.array_equal(import_and_compute('Add', [[1, 2, 3], [4, 5, 6]], [7, 8, 9]),
                          np.array([[8, 10, 12], [11, 13, 15]], dtype=np.float32))

    assert np.array_equal(import_and_compute('Add', [1, 2, 3], [[4, 5, 6], [7, 8, 9]]),
                          np.array([[5, 7, 9], [8, 10, 12]], dtype=np.float32))

    # shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
    left_operand = np.ones((2, 3, 4, 5)).astype(np.float32)
    assert np.array_equal(import_and_compute('Add', left_operand, 8), left_operand + 8)

    # shape(A) = (2, 3, 4, 5), shape(B) = (5,)
    left_operand = np.ones((2, 3, 4, 5), dtype=np.float32)
    right_operand = np.random.rand(5,).astype(np.float32)
    assert np.array_equal(import_and_compute('Add', left_operand, right_operand),
                          left_operand + right_operand)

    # shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
    left_operand = np.ones((2, 3, 4, 5), dtype=np.float32)
    right_operand = np.random.rand(4, 5).astype(np.float32)
    assert np.array_equal(import_and_compute('Add', left_operand, right_operand),
                          left_operand + right_operand)


@pytest.mark.skip(reason="ONNX spec not clear on this yet.")
def test_add_caffe_style():
    """Test Caffe2-style broadcasting with axis option

    Currently there is still discussion whether this explicit style of broadcasting should be
    enabled, or if it should be substituted by numpy-style implicit broadcasting.
    https://github.com/onnx/onnx/issues/83
    """
    # shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
    left_operand = np.zeros((2, 3, 4, 5), dtype=np.float32)
    right_operand = np.ones((3, 4), dtype=np.float32)
    node = make_node("Add", ["X", "Y"], ["Z"], name="test_node", broadcast=1, axis=1)
    graph = make_graph([node], "test_graph",
                       [make_tensor_value_info("X", onnx.TensorProto.FLOAT, left_operand.shape),
                        make_tensor_value_info("Y", onnx.TensorProto.FLOAT, right_operand.shape)],
                       [make_tensor_value_info("Z", onnx.TensorProto.FLOAT, left_operand.shape)])
    onnx_model = make_model(graph, producer_name='ngraph ONNXImporter')
    transformer = get_transformer()
    ng_model = import_onnx_model(onnx_model)[0]
    computation = transformer.computation(ng_model['output'], *ng_model['inputs'])
    assert np.array_equal(computation(left_operand, right_operand), np.ones((2, 3, 4, 5)))

    # shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
    left_operand = np.zeros((2, 3, 4, 5), dtype=np.float32)
    right_operand = np.ones((2,), dtype=np.float32)
    node = make_node("Add", ["X", "Y"], ["Z"], name="test_node", broadcast=1, axis=0)
    graph = make_graph([node], "test_graph",
                       [make_tensor_value_info("X", onnx.TensorProto.FLOAT, left_operand.shape),
                        make_tensor_value_info("Y", onnx.TensorProto.FLOAT, right_operand.shape)],
                       [make_tensor_value_info("Z", onnx.TensorProto.FLOAT, left_operand.shape)])
    onnx_model = make_model(graph, producer_name='ngraph ONNXImporter')
    transformer = get_transformer()
    ng_model = import_onnx_model(onnx_model)[0]
    computation = transformer.computation(ng_model['output'], *ng_model['inputs'])
    assert np.array_equal(computation(left_operand, right_operand), np.ones((2, 3, 4, 5)))


def test_sub():
    assert np.array_equal(import_and_compute('Sub', 20, 1),
                          np.array(19, dtype=np.float32))

    assert np.array_equal(import_and_compute('Sub', [20], [1]),
                          np.array([19], dtype=np.float32))

    assert np.array_equal(import_and_compute('Sub', [20, 19], [1, 2]),
                          np.array([19, 17], dtype=np.float32))

    assert np.array_equal(import_and_compute('Sub', [[1, 2, 3], [4, 5, 6]], [7, 8, 9]),
                          np.array([[-6, -6, -6], [-3, -3, -3]], dtype=np.float32))


def test_mul():
    assert np.array_equal(import_and_compute('Mul', 2, 3),
                          np.array(6, dtype=np.float32))

    assert np.array_equal(import_and_compute('Mul', [2], [3]),
                          np.array([6], dtype=np.float32))

    assert np.array_equal(import_and_compute('Mul', [2, 3], [4, 5]),
                          np.array([8, 15], dtype=np.float32))

    assert np.array_equal(import_and_compute('Mul', [[1, 2, 3], [4, 5, 6]], [7, 8, 9]),
                          np.array([[7, 16, 27], [28, 40, 54]], dtype=np.float32))


def test_div():
    assert np.array_equal(import_and_compute('Div', 6, 3),
                          np.array(2, dtype=np.float32))

    assert np.array_equal(import_and_compute('Div', [6], [3]),
                          np.array([2], dtype=np.float32))

    assert np.array_equal(import_and_compute('Div', [6, 8], [3, 2]),
                          np.array([2, 4], dtype=np.float32))

    assert np.array_equal(import_and_compute('Div', [[10, 20, 30], [40, 50, 60]], [2, 5, 6]),
                          np.array([[5, 4, 5], [20, 10, 10]], dtype=np.float32))
