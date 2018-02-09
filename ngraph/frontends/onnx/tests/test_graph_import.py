# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

from __future__ import print_function, division

import onnx
import ngraph as ng
import numpy as np
from onnx.helper import make_node, make_graph, make_tensor_value_info, make_model

from ngraph.frontends.onnx.onnx_importer.importer import import_onnx_model


def test_simple_graph():
    node1 = make_node('Add', ['A', 'B'], ['X'], name='add_node1')
    node2 = make_node('Add', ['X', 'C'], ['Y'], name='add_node2')
    graph = make_graph([node1, node2], 'test_graph',
                       [make_tensor_value_info('A', onnx.TensorProto.FLOAT, [1]),
                        make_tensor_value_info('B', onnx.TensorProto.FLOAT, [1]),
                        make_tensor_value_info('C', onnx.TensorProto.FLOAT, [1])],
                       [make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1])])
    model = make_model(graph, producer_name='ngraph ONNXImporter')

    ng_model = import_onnx_model(model)[0]
    transformer = ng.transformers.make_transformer()
    computation = transformer.computation(ng_model['output'], *ng_model['inputs'])
    assert np.array_equal(computation(4, 5, 6), np.array([15.0], dtype=np.float32))
