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
from functools import lru_cache

import onnx
import ngraph as ng
import numpy as np
from onnx.helper import make_node, make_graph, make_tensor_value_info, make_model
from ngraph.frontends.onnx.onnx_importer.importer import import_onnx_model


@lru_cache(maxsize=64)
def make_onnx_model_for_unary_op(op_type, input_shape, output_shape=None):
    if not output_shape:
        output_shape = input_shape

    node = make_node(op_type, ["X"], ["Y"], name="test_node")
    graph = make_graph([node], "test_graph",
                       [make_tensor_value_info("X", onnx.TensorProto.FLOAT, input_shape)],
                       [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, output_shape)])
    model = make_model(graph, producer_name='ngraph ONNXImporter')
    return model


@lru_cache(maxsize=1)
def get_transformer():
    return ng.transformers.make_transformer()


def import_model_make_computation(onnx_model):
    transformer = get_transformer()
    ng_model = import_onnx_model(onnx_model)[0]
    computation = transformer.computation(ng_model['output'], *ng_model['inputs'])
    return computation


def import_and_compute(op_type, input_data):
    input_data = np.array(input_data)
    model = make_onnx_model_for_unary_op(op_type, input_data.shape)
    computation = import_model_make_computation(model)
    return computation(input_data)


def test_abs():
    assert np.array_equal(import_and_compute('Abs', [-4, 0, 5, -10]),
                          np.array([4, 0, 5, 10], dtype=np.float32))

    assert np.array_equal(import_and_compute('Abs', [[-4, 0, 5, -10], [-4, 0, 5, -10]]),
                          np.array([[4, 0, 5, 10], [4, 0, 5, 10]], dtype=np.float32))

    assert np.array_equal(import_and_compute('Abs', [[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]),
                          np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]], dtype=np.float32))

