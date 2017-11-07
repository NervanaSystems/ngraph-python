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

from scipy.misc import logsumexp
from onnx.helper import make_node, make_graph, make_tensor_value_info, make_model
from ngraph.frontends.onnx.onnx_importer.importer import import_onnx_model


@lru_cache(maxsize=64)
def make_onnx_model_for_unary_op(op_type, input_shape, output_shape=None, **node_attrs):
    if not output_shape:
        output_shape = input_shape

    node = make_node(op_type, ["X"], ["Y"], name="test_node", **node_attrs)
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


def import_and_compute(op_type, input_data, **node_attrs):
    input_data = np.array(input_data)
    model = make_onnx_model_for_unary_op(op_type, input_data.shape, **node_attrs)
    computation = import_model_make_computation(model)
    return computation(input_data)


def test_abs():
    assert np.array_equal(import_and_compute('Abs', [-4, 0, 5, -10]),
                          np.array([4, 0, 5, 10], dtype=np.float32))

    assert np.array_equal(import_and_compute('Abs', [[-4, 0, 5, -10], [-4, 0, 5, -10]]),
                          np.array([[4, 0, 5, 10], [4, 0, 5, 10]], dtype=np.float32))

    assert np.array_equal(import_and_compute('Abs', [[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]),
                          np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]], dtype=np.float32))


def test_reduce_max():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute('ReduceMax', data, keepdims=0),
                          np.max(data, keepdims=False))
    assert np.array_equal(import_and_compute('ReduceMax', data, axes=(0,), keepdims=0),
                          np.max(data, keepdims=False, axis=(0,)))
    assert np.array_equal(import_and_compute('ReduceMax', data, axes=(1,), keepdims=0),
                          np.max(data, keepdims=False, axis=(1,)))
    assert np.array_equal(import_and_compute('ReduceMax', data, axes=(2,), keepdims=0),
                          np.max(data, keepdims=False, axis=(2,)))

    assert np.array_equal(import_and_compute('ReduceMax', data, axes=(0, 1), keepdims=0),
                          np.max(data, keepdims=False, axis=(0, 1)))
    assert np.array_equal(import_and_compute('ReduceMax', data, axes=(0, 2), keepdims=0),
                          np.max(data, keepdims=False, axis=(0, 2)))
    assert np.array_equal(import_and_compute('ReduceMax', data, axes=(1, 2), keepdims=0),
                          np.max(data, keepdims=False, axis=(1, 2)))

    assert np.array_equal(import_and_compute('ReduceMax', data, axes=(0, 1, 2), keepdims=0),
                          np.max(data, keepdims=False, axis=(0, 1, 2)))


def test_reduce_max_keepdims():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute('ReduceMax', data), np.max(data, keepdims=True))
    assert np.array_equal(import_and_compute('ReduceMax', data, axes=(0,)),
                          np.max(data, keepdims=True, axis=(0,)))
    assert np.array_equal(import_and_compute('ReduceMax', data, axes=(1,)),
                          np.max(data, keepdims=True, axis=(1,)))
    assert np.array_equal(import_and_compute('ReduceMax', data, axes=(2,)),
                          np.max(data, keepdims=True, axis=(2,)))

    assert np.array_equal(import_and_compute('ReduceMax', data, axes=(0, 1)),
                          np.max(data, keepdims=True, axis=(0, 1)))
    assert np.array_equal(import_and_compute('ReduceMax', data, axes=(0, 2)),
                          np.max(data, keepdims=True, axis=(0, 2)))
    assert np.array_equal(import_and_compute('ReduceMax', data, axes=(1, 2)),
                          np.max(data, keepdims=True, axis=(1, 2)))

    assert np.array_equal(import_and_compute('ReduceMax', data, axes=(0, 1, 2)),
                          np.max(data, keepdims=True, axis=(0, 1, 2)))


def test_reduce_min():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute('ReduceMin', data), np.min(data, keepdims=True))
    assert np.array_equal(import_and_compute('ReduceMin', data, keepdims=0),
                          np.min(data, keepdims=False))

    assert np.array_equal(import_and_compute('ReduceMin', data, axes=(1,)),
                          np.min(data, keepdims=True, axis=(1,)))
    assert np.array_equal(import_and_compute('ReduceMin', data, axes=(1,), keepdims=0),
                          np.min(data, keepdims=False, axis=(1,)))

    assert np.array_equal(import_and_compute('ReduceMin', data, axes=(0, 2)),
                          np.min(data, keepdims=True, axis=(0, 2)))
    assert np.array_equal(import_and_compute('ReduceMin', data, axes=(0, 2), keepdims=0),
                          np.min(data, keepdims=False, axis=(0, 2)))

    assert np.array_equal(import_and_compute('ReduceMin', data, axes=(0, 1, 2)),
                          np.min(data, keepdims=True, axis=(0, 1, 2)))
    assert np.array_equal(import_and_compute('ReduceMin', data, axes=(0, 1, 2), keepdims=0),
                          np.min(data, keepdims=False, axis=(0, 1, 2)))


def test_reduce_mean():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute('ReduceMean', data), np.mean(data, keepdims=True))
    assert np.array_equal(import_and_compute('ReduceMean', data, keepdims=0),
                          np.mean(data, keepdims=False))

    assert np.array_equal(import_and_compute('ReduceMean', data, axes=(1,)),
                          np.mean(data, keepdims=True, axis=(1,)))
    assert np.array_equal(import_and_compute('ReduceMean', data, axes=(1,), keepdims=0),
                          np.mean(data, keepdims=False, axis=(1,)))

    assert np.array_equal(import_and_compute('ReduceMean', data, axes=(0, 2)),
                          np.mean(data, keepdims=True, axis=(0, 2)))
    assert np.array_equal(import_and_compute('ReduceMean', data, axes=(0, 2), keepdims=0),
                          np.mean(data, keepdims=False, axis=(0, 2)))

    assert np.array_equal(import_and_compute('ReduceMean', data, axes=(0, 1, 2)),
                          np.mean(data, keepdims=True, axis=(0, 1, 2)))
    assert np.array_equal(import_and_compute('ReduceMean', data, axes=(0, 1, 2), keepdims=0),
                          np.mean(data, keepdims=False, axis=(0, 1, 2)))


def test_reduce_sum():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute('ReduceSum', data), np.sum(data, keepdims=True))
    assert np.array_equal(import_and_compute('ReduceSum', data, keepdims=0),
                          np.sum(data, keepdims=False))

    assert np.array_equal(import_and_compute('ReduceSum', data, axes=(1,)),
                          np.sum(data, keepdims=True, axis=(1,)))
    assert np.array_equal(import_and_compute('ReduceSum', data, axes=(1,), keepdims=0),
                          np.sum(data, keepdims=False, axis=(1,)))

    assert np.array_equal(import_and_compute('ReduceSum', data, axes=(0, 2)),
                          np.sum(data, keepdims=True, axis=(0, 2)))
    assert np.array_equal(import_and_compute('ReduceSum', data, axes=(0, 2), keepdims=0),
                          np.sum(data, keepdims=False, axis=(0, 2)))

    assert np.array_equal(import_and_compute('ReduceSum', data, axes=(0, 1, 2)),
                          np.sum(data, keepdims=True, axis=(0, 1, 2)))
    assert np.array_equal(import_and_compute('ReduceSum', data, axes=(0, 1, 2), keepdims=0),
                          np.sum(data, keepdims=False, axis=(0, 1, 2)))


def test_reduce_prod():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute('ReduceProd', data), np.prod(data, keepdims=True))
    assert np.array_equal(import_and_compute('ReduceProd', data, keepdims=0),
                          np.prod(data, keepdims=False))

    assert np.array_equal(import_and_compute('ReduceProd', data, axes=(1,)),
                          np.prod(data, keepdims=True, axis=(1,)))
    assert np.array_equal(import_and_compute('ReduceProd', data, axes=(1,), keepdims=0),
                          np.prod(data, keepdims=False, axis=(1,)))

    assert np.array_equal(import_and_compute('ReduceProd', data, axes=(0, 2)),
                          np.prod(data, keepdims=True, axis=(0, 2)))
    assert np.array_equal(import_and_compute('ReduceProd', data, axes=(0, 2), keepdims=0),
                          np.prod(data, keepdims=False, axis=(0, 2)))

    assert np.array_equal(import_and_compute('ReduceProd', data, axes=(0, 1, 2)),
                          np.prod(data, keepdims=True, axis=(0, 1, 2)))
    assert np.array_equal(import_and_compute('ReduceProd', data, axes=(0, 1, 2), keepdims=0),
                          np.prod(data, keepdims=False, axis=(0, 1, 2)))


def test_reduce_log_sum_exp():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute('ReduceLogSumExp', data),
                          logsumexp(data, keepdims=True))
    assert np.array_equal(import_and_compute('ReduceLogSumExp', data, keepdims=0),
                          logsumexp(data, keepdims=False))

    assert np.array_equal(import_and_compute('ReduceLogSumExp', data, axes=(1,)),
                          logsumexp(data, keepdims=True, axis=(1,)))
    assert np.array_equal(import_and_compute('ReduceLogSumExp', data, axes=(1,), keepdims=0),
                          logsumexp(data, keepdims=False, axis=(1,)))

    assert np.array_equal(import_and_compute('ReduceLogSumExp', data, axes=(0, 2)),
                          logsumexp(data, keepdims=True, axis=(0, 2)))
    assert np.array_equal(import_and_compute('ReduceLogSumExp', data, axes=(0, 2), keepdims=0),
                          logsumexp(data, keepdims=False, axis=(0, 2)))

    assert np.array_equal(import_and_compute('ReduceLogSumExp', data, axes=(0, 1, 2)),
                          logsumexp(data, keepdims=True, axis=(0, 1, 2)))
    assert np.array_equal(import_and_compute('ReduceLogSumExp', data, axes=(0, 1, 2), keepdims=0),
                          logsumexp(data, keepdims=False, axis=(0, 1, 2)))
