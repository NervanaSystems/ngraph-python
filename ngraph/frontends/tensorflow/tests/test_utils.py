# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
"""
Unit tests for ngraph/frontends/tensorflow/tf_importer/utils
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ngraph as ng
import pytest
from ngraph.frontends.tensorflow.tf_importer.utils import np_layout_shuffle
from ngraph.frontends.tensorflow.tf_importer.utils_broadcast import \
    broadcast_to, is_compatible_numpy_shape, is_compatible_broadcast_shape, \
    broadcasted_shape
from ngraph.frontends.tensorflow.tf_importer.utils_pos_axes import make_pos_axes
from ngraph.testing.execution import ExecutorFactory


def test_np_layout_shuffle():
    # set up
    bsz = 8
    C, H, W, N = 3, 28, 28, bsz
    C, R, S, K = 3, 5, 5, 32

    # image dim-shuffle
    np_tf_image = np.random.randn(N, H, W, C)
    np_ng_image = np_layout_shuffle(np_tf_image, "NHWC", "CDHWN")
    np_tf_image_reverse = np_layout_shuffle(np_ng_image, "CDHWN", "NHWC")
    assert np.array_equal(np_tf_image, np_tf_image_reverse)

    # filter dim-shuffle
    np_tf_weight = np.random.randn(R, S, C, K)
    np_ng_weight = np_layout_shuffle(np_tf_weight, "RSCK", "CTRSK")
    np_tf_weight_reverse = np_layout_shuffle(np_ng_weight, "CTRSK", "RSCK")
    assert np.array_equal(np_tf_weight, np_tf_weight_reverse)


@pytest.mark.parametrize("test_case", [
    [(), (), True],
    [(), (1, 2, 3), True],
    [(1, 2), (2, 2), True],
    [(3, 2), (2, 2), False],
    [(2, 1, 2, 1), (1, 1, 3), True],
    [(2, 1, 2, 1), (2, 1, 3), True],
    [(2, 1, 2, 1), (1, 3, 1), False],
])
def test_is_compatible_numpy_shape(test_case):
    left_shape, right_shape, result = test_case
    assert is_compatible_numpy_shape(left_shape, right_shape) == result


@pytest.mark.parametrize("test_case", [
    [(), (), True],
    [(), (1, 2, 3), True],
    [(1, 2, 3), (), False],
    [(1, 2), (2, 2), True],
    [(2, 2), (1, 2), False],
    [(3, 2), (2, 2), False],
    [(2, 1, 2, 1), (1, 1, 3), False],
    [(2, 1, 3), (2, 1, 2, 1), False],
    [(2, 1, 3), (4, 2, 2, 3), True],
    [(2, 1, 3), (4, 2, 2, 5), False]
])
def test_is_compatible_broadcast_shape(test_case):
    left_shape, right_shape, result = test_case
    assert is_compatible_broadcast_shape(left_shape, right_shape) == result


@pytest.mark.transformer_dependent
@pytest.mark.parametrize("test_case", [
    [(), (1,)],
    [(), (1, 2)],
    [(1,), (2,)],
    [(1,), (2, 1)],
    [(1,), (3, 2)],
    [(2,), (3, 2)],
    [(1, 3, 1), (2, 3, 4)],
    [(3, 1, 2), (4, 3, 5, 2)],
    [(5, 1, 2, 1), (4, 5, 1, 2, 3)],
])
def test_broadcast_to(test_case):
    src_shape, dst_shape = test_case

    # numpy results
    x_np = np.array(np.random.rand(*src_shape))
    f_np = x_np + np.zeros(dst_shape)

    # ngraph results
    x_ng = ng.constant(x_np, axes=make_pos_axes(x_np.shape))
    f_ng = broadcast_to(x_ng, dst_shape)

    with ExecutorFactory() as ex:
        f_ng_comp = ex.transformer.computation(f_ng)
        f_ng_val = f_ng_comp()
        np.testing.assert_allclose(f_ng_val, f_np)


@pytest.mark.parametrize("test_case", [
    [(), (), ()],
    [(), (1,), (1,)],
    [(1,), (), (1,)],
    [(), (1, 2), (1, 2)],
    [(1, 2), (), (1, 2)],
    [(), (1, 2, 3), (1, 2, 3)],
    [(1, 2), (2, 2), (2, 2)],
    [(2, 1, 3), (4, 2, 2, 3), (4, 2, 2, 3)],
])
def test_broadcasted_shape(test_case):
    left_shape, right_shape, out_shape = test_case
    assert (broadcasted_shape(left_shape, right_shape) == out_shape)
