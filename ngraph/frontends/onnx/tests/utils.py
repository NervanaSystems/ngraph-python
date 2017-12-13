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

import numpy as np
import onnx
from onnx.helper import make_graph, make_model, make_tensor_value_info

import ngraph as ng
from ngraph.frontends.onnx.onnx_importer.importer import import_onnx_model


def get_transformer():
    return ng.transformers.make_transformer()


def convert_and_calculate(onnx_node, data_inputs, data_outputs):
    # type: (NodeProto, List[np.ndarray], List[np.ndarray]) -> List[np.ndarray]
    """
    Convert ONNX node to ngraph node and perform computation on input data.

    :param onnx_node: ONNX NodeProto describing a computation node
    :param data_inputs: list of numpy ndarrays with input data
    :param data_outputs: list of numpy ndarrays with expected output data
    :return: list of numpy ndarrays with computed output
    """
    transformer = get_transformer()
    input_tensors = [make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape)
                     for name, value in zip(onnx_node.input, data_inputs)]
    output_tensors = [make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape)
                      for name, value in zip(onnx_node.output, data_outputs)]

    graph = make_graph([onnx_node], 'test_graph', input_tensors, output_tensors)
    model = make_model(graph, producer_name='ngraph ONNXImporter')

    ng_results = []
    for ng_model in import_onnx_model(model):
        computation = transformer.computation(ng_model['output'], *ng_model['inputs'])
        ng_results.append(computation(*data_inputs))

    return ng_results


def all_arrays_equal(first_list, second_list):
    # type: (Iterable[np.ndarray], Iterable[np.ndarray]) -> bool
    """
    Check that all numpy ndarrays in `first_list` are equal to all numpy ndarrays in `second_list`.

    :param first_list: iterable containing numpy ndarray objects
    :param second_list: another iterable containing numpy ndarray objects
    :return: True if all ndarrays are equal, otherwise False
    """
    return all(map(lambda pair: np.array_equal(*pair), zip(first_list, second_list)))
