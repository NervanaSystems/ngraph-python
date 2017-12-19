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

import onnx
import numpy as np

from scipy.misc import logsumexp
from ngraph.frontends.onnx.tests.utils import convert_and_calculate


def import_and_compute(op_type, input_data, **node_attrs):
    data_inputs = [np.array(input_data)]
    node = onnx.helper.make_node(op_type, inputs=['x'], outputs=['y'], **node_attrs)
    return convert_and_calculate(node, data_inputs, data_inputs).pop()


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


def test_reduce_argmin():
    def argmin(ndarray, axis, keepdims=False):
        res = np.argmin(ndarray, axis=axis)
        if keepdims:
            res = np.expand_dims(res, axis=axis)
        return res

    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute('ArgMin', data, axis=0),
                          argmin(data, keepdims=True, axis=0))
    assert np.array_equal(import_and_compute('ArgMin', data, axis=0, keepdims=0),
                          argmin(data, keepdims=False, axis=0))
    assert np.array_equal(import_and_compute('ArgMin', data, axis=1),
                          argmin(data, keepdims=True, axis=1))
    assert np.array_equal(import_and_compute('ArgMin', data, axis=1, keepdims=0),
                          argmin(data, keepdims=False, axis=1))
    assert np.array_equal(import_and_compute('ArgMin', data, axis=2),
                          argmin(data, keepdims=True, axis=2))
    assert np.array_equal(import_and_compute('ArgMin', data, axis=2, keepdims=0),
                          argmin(data, keepdims=False, axis=2))


def test_reduce_argmax():
    def argmax(ndarray, axis, keepdims=False):
        res = np.argmax(ndarray, axis=axis)
        if keepdims:
            res = np.expand_dims(res, axis=axis)
        return res

    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute('ArgMax', data, axis=0),
                          argmax(data, keepdims=True, axis=0))
    assert np.array_equal(import_and_compute('ArgMax', data, axis=0, keepdims=0),
                          argmax(data, keepdims=False, axis=0))
    assert np.array_equal(import_and_compute('ArgMax', data, axis=1),
                          argmax(data, keepdims=True, axis=1))
    assert np.array_equal(import_and_compute('ArgMax', data, axis=1, keepdims=0),
                          argmax(data, keepdims=False, axis=1))
    assert np.array_equal(import_and_compute('ArgMax', data, axis=2),
                          argmax(data, keepdims=True, axis=2))
    assert np.array_equal(import_and_compute('ArgMax', data, axis=2, keepdims=0),
                          argmax(data, keepdims=False, axis=2))
