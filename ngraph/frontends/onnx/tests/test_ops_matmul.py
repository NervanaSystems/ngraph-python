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

import ngraph as ng
import numpy as np
from onnx.helper import make_node, make_graph, make_tensor_value_info, make_model
from ngraph.frontends.onnx.onnx_importer.importer import import_onnx_model


@lru_cache(maxsize=1)
def get_transformer():
    return ng.transformers.make_transformer()


def make_onnx_model_for_dot_op(input_left, input_right):
    output_shape = np.dot(input_left, input_right).shape
    node = make_node('Dot', ["X", "Y"], ["Z"], name="test_node")
    graph = make_graph([node], "test_graph",
                       [make_tensor_value_info("X", onnx.TensorProto.FLOAT, input_left.shape),
                        make_tensor_value_info("Y", onnx.TensorProto.FLOAT, input_right.shape)],
                       [make_tensor_value_info("Z", onnx.TensorProto.FLOAT, output_shape)])
    model = make_model(graph, producer_name='ngraph ONNXImporter')
    return model


def import_and_compute_dot(input_left, input_right):
    input_data_left = np.array(input_left)
    input_data_right = np.array(input_right)
    onnx_model = make_onnx_model_for_dot_op(input_data_left, input_data_right)
    transformer = get_transformer()
    ng_model = import_onnx_model(onnx_model)[0]
    computation = transformer.computation(ng_model['output'], *ng_model['inputs'])
    return computation(input_data_left, input_data_right)


def numpy_gemm(input_a, input_b, input_c, alpha=1, beta=1, trans_a=False, trans_b=False):
    input_a, input_b, input_c = np.array(input_a), np.array(input_b), np.array(input_c)
    if trans_a:
        input_a = input_a.T
    if trans_b:
        input_b = input_b.T

    return (alpha * np.dot(input_a, input_b)) + (beta * input_c)


def make_onnx_model_for_gemm_op(input_a, input_b, input_c, **kwargs):
    output_shape = np.dot(input_a, input_b).shape
    node = make_node('Gemm', ["A", "B", "C"], ["Y"], name="test_node", **kwargs)
    graph = make_graph([node], "test_graph",
                       [make_tensor_value_info("A", onnx.TensorProto.FLOAT, input_a.shape),
                        make_tensor_value_info("B", onnx.TensorProto.FLOAT, input_b.shape),
                        make_tensor_value_info("C", onnx.TensorProto.FLOAT, input_c.shape)],
                       [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, output_shape)])
    model = make_model(graph, producer_name='ngraph ONNXImporter')
    return model


def import_and_compute_gemm(input_a, input_b, input_c, **kwargs):
    input_a, input_b, input_c = np.array(input_a), np.array(input_b), np.array(input_c)

    if kwargs.get('trans_a'):
        kwargs['transA'] = kwargs['trans_a']
        del kwargs['trans_a']

    if kwargs.get('trans_b'):
        kwargs['transB'] = kwargs['trans_b']
        del kwargs['trans_b']

    onnx_model = make_onnx_model_for_gemm_op(input_a, input_b, input_c, **kwargs)
    transformer = get_transformer()
    ng_model = import_onnx_model(onnx_model)[0]
    computation = transformer.computation(ng_model['output'], *ng_model['inputs'])
    return computation(input_a, input_b, input_c)


def test_dot():
    # vector @ vector
    data = ([1, 2], [1, 3])
    assert np.array_equal(import_and_compute_dot(*data), np.dot(*data))

    data = ([1, 2, 3], [[4], [5], [6]])
    assert np.array_equal(import_and_compute_dot(*data), np.dot(*data))

    data = ([[1, 2, 3]], [1, 2, 3])
    assert np.array_equal(import_and_compute_dot(*data), np.dot(*data))

    # vector @ matrix
    data = ([1, 2, 3], [[4, 5], [6, 7], [8, 9]])
    assert np.array_equal(import_and_compute_dot(*data), np.dot(*data))

    # matrix @ vector
    data = ([[1, 2, 3], [4, 5, 6]], [[7], [8], [9]])
    assert np.array_equal(import_and_compute_dot(*data), np.dot(*data))

    # matrix @ matrix
    data = ([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    assert np.array_equal(import_and_compute_dot(*data), np.dot(*data))

    data = ([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])
    assert np.array_equal(import_and_compute_dot(*data), np.dot(*data))

    data = ([[1, 2], [3, 4], [5, 6]], [[7, 8, 9], [10, 11, 12]])
    assert np.array_equal(import_and_compute_dot(*data), np.dot(*data))

    # 3d tensor @ 3d tensor
    data = ([[[1, 2], [3, 4]], [[1, 2], [3, 4]]], [[[5, 6], [7, 8]], [[5, 6], [7, 8]]])
    assert np.array_equal(import_and_compute_dot(*data), np.dot(*data))

    data = (np.ones((5, 2, 3)), (np.ones((5, 3, 2)) + 2))
    assert np.array_equal(import_and_compute_dot(*data), np.dot(*data))


def test_gemm():
    data = ([1, 2], [1, 3], [1, 4])
    assert np.array_equal(import_and_compute_gemm(*data), numpy_gemm(*data))

    data = ([1, 2], [1, 3], [1, 4])
    kwargs = {'trans_a': True, 'trans_b': True}
    assert np.array_equal(import_and_compute_gemm(*data, **kwargs), numpy_gemm(*data, **kwargs))

    data = ([1, 2], [1, 3], [1, 4])
    kwargs = {'alpha': 7, 'beta': 9}
    assert np.array_equal(import_and_compute_gemm(*data, **kwargs), numpy_gemm(*data, **kwargs))

    data = ([[1, 2], [1, 2]], [[1, 3], [1, 3]], [4, 1])
    kwargs = {'trans_a': True, 'trans_b': True, 'alpha': 7, 'beta': 9}
    assert np.array_equal(import_and_compute_gemm(*data, **kwargs), numpy_gemm(*data, **kwargs))
