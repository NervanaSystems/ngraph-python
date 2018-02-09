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

import cntk as C
import numpy as np

import ngraph as ng
from ngraph.frontends.cntk.cntk_importer.importer import CNTKImporter


def assert_cntk_ngraph_isclose(cntk_op):
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.isclose(cntk_ret, ng_ret).all()


def assert_cntk_ngraph_array_equal(cntk_op):
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def assert_cntk_ngraph_flat_equal(cntk_op):
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret.flatten(), ng_ret.flatten())


def assert_cntk_ngraph_flat_isclose(cntk_op):
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.isclose(cntk_ret.flatten(), ng_ret.flatten()).all()


def test_sigmoid():
    assert_cntk_ngraph_isclose(C.sigmoid([-2, -1., 0., 1., 2.]))
    assert_cntk_ngraph_isclose(C.sigmoid([0.]))
    assert_cntk_ngraph_isclose(C.exp([-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.]))


def test_exp():
    assert_cntk_ngraph_isclose(C.exp([-2, -1., 0., 1., 2.]))
    assert_cntk_ngraph_isclose(C.exp([0.]))
    assert_cntk_ngraph_isclose(C.exp([-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.]))


def test_tanh():
    assert_cntk_ngraph_isclose(C.tanh([-2, -1., 0., 1., 2.]))
    assert_cntk_ngraph_isclose(C.tanh([0.]))
    assert_cntk_ngraph_isclose(C.tanh([-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.]))


def test_relu():
    assert_cntk_ngraph_array_equal(C.relu([-2, -1., 0., 1., 2.]))
    assert_cntk_ngraph_array_equal(C.relu([0.]))
    assert_cntk_ngraph_array_equal(C.relu([-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1]))
    assert_cntk_ngraph_array_equal(C.relu([[1, 2, 3], [4, 5, 6]]))
    assert_cntk_ngraph_array_equal(C.relu([[-3, -2, -1], [1, 2, 3]]))


def test_reciprocal():
    assert_cntk_ngraph_isclose(C.reciprocal([-1 / 3, 1 / 5, -2, 3]))
    assert_cntk_ngraph_isclose(C.reciprocal([[-1, 0.5], [-3, 4]]))
    assert_cntk_ngraph_isclose(C.reciprocal([[[1, 0.5], [-3, 0.33]], [[1, -2], [3, 4]]]))


def test_negate():
    assert_cntk_ngraph_array_equal(C.negate([-1, 1, -2, 3]))
    assert_cntk_ngraph_array_equal(C.negate([[-1, 0], [3, -4]]))
    assert_cntk_ngraph_array_equal(C.negate([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]))


def test_log():
    assert_cntk_ngraph_array_equal(C.log([1., 2.]))
    assert_cntk_ngraph_array_equal(C.log([[1, 2], [3, 4]]))
    assert_cntk_ngraph_array_equal(C.log([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]))


def test_sqrt():
    assert_cntk_ngraph_isclose(C.sqrt([0., 4.]))
    assert_cntk_ngraph_isclose(C.sqrt([[1, 2], [3, 4]]))
    assert_cntk_ngraph_isclose(C.sqrt([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]))


def test_floor():
    assert_cntk_ngraph_array_equal(C.floor([0.2, 1.3, 4., 5.5, 0.0]))
    assert_cntk_ngraph_array_equal(C.floor([[0.6, 3.3], [1.9, 5.6]]))
    assert_cntk_ngraph_array_equal(C.floor([-5.5, -4.2, -3., -0.7, 0]))
    assert_cntk_ngraph_array_equal(C.floor([[-0.6, -4.3], [1.9, -3.2]]))


def test_abs():
    assert_cntk_ngraph_array_equal(C.abs([-1, 1, -2, 3]))
    assert_cntk_ngraph_array_equal(C.abs([[1, -2], [3, -4]]))
    assert_cntk_ngraph_array_equal(C.abs([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]))


def test_softmax():
    assert_cntk_ngraph_isclose(C.softmax([[1, 1, 2, 3]]))
    assert_cntk_ngraph_isclose(C.softmax([1, 1]))
    assert_cntk_ngraph_isclose(C.softmax([[[1, 1], [3, 5]]], axis=-1))
    # This test is failing, bug must be fixed:
    # assert_cntk_ngraph_isclose(C.softmax([[[1, 1], [3, 5]]], axis=1))


def test_reduce_max():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    # This test is failing, bug must be fixed:
    # assert_cntk_ngraph_flat_equal(C.reduce_max([1, 0]))
    assert_cntk_ngraph_flat_equal(C.reduce_max([1, 0], 0))
    assert_cntk_ngraph_flat_equal(C.reduce_max([[1., 1.], [3., 5.]], 0))
    assert_cntk_ngraph_flat_equal(C.reduce_max([[1., 1.], [3., 5.]], 1))
    assert_cntk_ngraph_flat_equal(C.reduce_max([[1., 1.], [3., 5.]], -1))
    assert_cntk_ngraph_flat_equal(C.reduce_max(data, 0))
    assert_cntk_ngraph_flat_equal(C.reduce_max(data, 1))
    assert_cntk_ngraph_flat_equal(C.reduce_max(data, 2))
    assert_cntk_ngraph_flat_equal(C.reduce_max(data, -1))
    assert_cntk_ngraph_flat_equal(C.reduce_max(data, (0, 1)))
    assert_cntk_ngraph_flat_equal(C.reduce_max(data, (0, 2)))
    assert_cntk_ngraph_flat_equal(C.reduce_max(data, (1, 2)))
    assert_cntk_ngraph_flat_equal(C.reduce_max(data, (-1, -2)))


def test_reduce_min():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert_cntk_ngraph_flat_equal(C.reduce_min([1, 0], 0))
    assert_cntk_ngraph_flat_equal(C.reduce_min([[1., 1.], [3., 5.]], 0))
    assert_cntk_ngraph_flat_equal(C.reduce_min([[1., 1.], [3., 5.]], 1))
    assert_cntk_ngraph_flat_equal(C.reduce_min([[1., 1.], [3., 5.]], -1))
    assert_cntk_ngraph_flat_equal(C.reduce_min(data, 0))
    assert_cntk_ngraph_flat_equal(C.reduce_min(data, 1))
    assert_cntk_ngraph_flat_equal(C.reduce_min(data, 2))
    assert_cntk_ngraph_flat_equal(C.reduce_min(data, -1))
    assert_cntk_ngraph_flat_equal(C.reduce_min(data, (0, 1)))
    assert_cntk_ngraph_flat_equal(C.reduce_min(data, (0, 2)))
    assert_cntk_ngraph_flat_equal(C.reduce_min(data, (1, 2)))
    assert_cntk_ngraph_flat_equal(C.reduce_min(data, (-1, -2)))


def test_reduce_sum():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert_cntk_ngraph_flat_equal(C.reduce_sum([1, 0], 0))
    assert_cntk_ngraph_flat_equal(C.reduce_sum([[1., 1.], [3., 5.]], 0))
    assert_cntk_ngraph_flat_equal(C.reduce_sum([[1., 1.], [3., 5.]], 1))
    assert_cntk_ngraph_flat_equal(C.reduce_sum([[1., 1.], [3., 5.]], -1))
    assert_cntk_ngraph_flat_equal(C.reduce_sum(data, 0))
    assert_cntk_ngraph_flat_equal(C.reduce_sum(data, 1))
    assert_cntk_ngraph_flat_equal(C.reduce_sum(data, 2))
    assert_cntk_ngraph_flat_equal(C.reduce_sum(data, -1))
    assert_cntk_ngraph_flat_equal(C.reduce_sum(data, (0, 1)))
    assert_cntk_ngraph_flat_equal(C.reduce_sum(data, (0, 2)))
    assert_cntk_ngraph_flat_equal(C.reduce_sum(data, (1, 2)))
    assert_cntk_ngraph_flat_equal(C.reduce_sum(data, (-1, -2)))


def test_reduce_mean():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert_cntk_ngraph_flat_equal(C.reduce_mean([1, 0], 0))
    assert_cntk_ngraph_flat_equal(C.reduce_mean([[1., 1.], [3., 5.]], 0))
    assert_cntk_ngraph_flat_equal(C.reduce_mean([[1., 1.], [3., 5.]], 1))
    assert_cntk_ngraph_flat_equal(C.reduce_mean([[1., 1.], [3., 5.]], -1))
    assert_cntk_ngraph_flat_equal(C.reduce_mean(data, 0))
    assert_cntk_ngraph_flat_equal(C.reduce_mean(data, 1))
    assert_cntk_ngraph_flat_equal(C.reduce_mean(data, 2))
    assert_cntk_ngraph_flat_equal(C.reduce_mean(data, -1))
    assert_cntk_ngraph_flat_equal(C.reduce_mean(data, (0, 1)))
    assert_cntk_ngraph_flat_equal(C.reduce_mean(data, (0, 2)))
    assert_cntk_ngraph_flat_equal(C.reduce_mean(data, (1, 2)))
    assert_cntk_ngraph_flat_equal(C.reduce_mean(data, (-1, -2)))


def test_reduce_prod():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert_cntk_ngraph_flat_equal(C.reduce_prod([1, 0], 0))
    assert_cntk_ngraph_flat_equal(C.reduce_prod([[1., 1.], [3., 5.]], 0))
    assert_cntk_ngraph_flat_equal(C.reduce_prod([[1., 1.], [3., 5.]], 1))
    assert_cntk_ngraph_flat_equal(C.reduce_prod([[1., 1.], [3., 5.]], -1))
    assert_cntk_ngraph_flat_equal(C.reduce_prod(data, 0))
    assert_cntk_ngraph_flat_equal(C.reduce_prod(data, 1))
    assert_cntk_ngraph_flat_equal(C.reduce_prod(data, 2))
    assert_cntk_ngraph_flat_equal(C.reduce_prod(data, -1))
    assert_cntk_ngraph_flat_equal(C.reduce_prod(data, (0, 1)))
    assert_cntk_ngraph_flat_equal(C.reduce_prod(data, (0, 2)))
    assert_cntk_ngraph_flat_equal(C.reduce_prod(data, (1, 2)))
    assert_cntk_ngraph_flat_equal(C.reduce_prod(data, (-1, -2)))


def test_reduce_log_sum_exp():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert_cntk_ngraph_flat_isclose(C.reduce_log_sum_exp([1, 0], 0))
    assert_cntk_ngraph_flat_isclose(C.reduce_log_sum_exp([[1., 1.], [3., 5.]], 0))
    assert_cntk_ngraph_flat_isclose(C.reduce_log_sum_exp([[1., 1.], [3., 5.]], 1))
    assert_cntk_ngraph_flat_isclose(C.reduce_log_sum_exp([[1., 1.], [3., 5.]], -1))
    assert_cntk_ngraph_flat_isclose(C.reduce_log_sum_exp(data, 0))
    assert_cntk_ngraph_flat_isclose(C.reduce_log_sum_exp(data, 1))
    assert_cntk_ngraph_flat_isclose(C.reduce_log_sum_exp(data, 2))
    assert_cntk_ngraph_flat_isclose(C.reduce_log_sum_exp(data, -1))
    assert_cntk_ngraph_flat_isclose(C.reduce_log_sum_exp(data, (0, 1)))
    assert_cntk_ngraph_flat_isclose(C.reduce_log_sum_exp(data, (0, 2)))
    assert_cntk_ngraph_flat_isclose(C.reduce_log_sum_exp(data, (1, 2)))
    assert_cntk_ngraph_flat_isclose(C.reduce_log_sum_exp(data, (-1, -2)))


if __name__ == "__main__":
    test_sigmoid()
    test_exp()
    test_tanh()
    test_relu()
    test_reciprocal()
    test_negate()
    test_log()
    test_sqrt()
    test_floor()
    test_abs()
    test_softmax()
    test_reduce_max()
    test_reduce_min()
    test_reduce_mean()
    test_reduce_sum()
    test_reduce_prod()
    test_reduce_log_sum_exp()
