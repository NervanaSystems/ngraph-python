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
from __future__ import division
import numpy as np
import pytest
import ngraph as ng

from ngraph.testing import executor, ExecutorFactory

pytestmark = pytest.mark.transformer_dependent


@pytest.fixture(scope='module', params=[
    (64, 10, 28, 32),
    (3, 16, 16, 4),
    (7, 15, 19, 5),
    (3, 5, 7, 2)])
def input_axes(request):
    return ng.make_axes([
        ng.make_axis(length=request.param[0]),
        ng.make_axis(length=request.param[1]),
        ng.make_axis(length=request.param[2]),
        ng.make_axis(length=request.param[3])
    ])


def test_exit_condition():
    bsz = 16
    class_num = 10

    N, Y = ng.make_axis(bsz), ng.make_axis(class_num)
    y_val = np.absolute(np.random.randn(bsz, class_num))
    y = ng.constant(y_val, ng.make_axes([N, Y]))

    likelihood = ng.log(ng.softmax(y, normalization_axes=y.axes[1]))

    with ExecutorFactory() as ex:
        comp = ex.executor(likelihood)

        val1 = comp()
        val2 = comp()
        np.testing.assert_allclose(val1, val2, atol=0, rtol=0)


def test_4d_elementwise(input_axes):
    x_val = np.absolute(np.random.randn(*input_axes.lengths))
    y_val = np.absolute(np.random.randn(*input_axes.lengths))
    x = ng.constant(x_val, input_axes)
    y = ng.constant(y_val, input_axes)

    out = ng.add(x, y)

    with executor(out) as ex:
        graph_val = ex()
    np_val = np.add(x_val, y_val)
    ng.testing.assert_allclose(graph_val, np_val, rtol=1e-4)


def test_4d_reduction(input_axes):
    x_val = np.absolute(np.random.randn(*input_axes.lengths))
    x = ng.constant(x_val, input_axes)

    out1 = ng.sum(x, reduction_axes=input_axes[1])
    out2 = ng.sum(x, reduction_axes=input_axes[3])

    with executor([out1, out2]) as ex:
        graph_val1, graph_val2 = ex()
        np_val1 = np.sum(x_val, 1)
        np_val2 = np.sum(x_val, 3)
        ng.testing.assert_allclose(graph_val1, np_val1, rtol=1e-4)
        ng.testing.assert_allclose(graph_val2, np_val2, rtol=1e-4)


@pytest.config.flex_skip(reason="Fail for flex, but randomly passing, due to random input -> SKIP")
def test_4d_chained(input_axes):
    x_val = np.absolute(np.random.randn(*input_axes.lengths))
    y_val = np.absolute(np.random.randn(*input_axes.lengths))
    x = ng.constant(x_val, input_axes)
    y = ng.constant(y_val, input_axes)

    im = ng.reciprocal(x)
    out = ng.sum(ng.add(im, y), reduction_axes=input_axes[0])

    with executor(out) as ex:
        graph_val = ex()
    np_val = np.sum(np.add(np.reciprocal(x_val), y_val), 0)
    ng.testing.assert_allclose(graph_val, np_val, rtol=1e-4)


@pytest.config.flex_disabled(reason='Results mismatch')
def test_kernel_cache():
    X = ng.make_axis(32)
    Y = ng.make_axis(32)
    C = ng.make_axis(16384)
    axes = ng.make_axes([
        X,
        Y
    ])
    bcast_axes = ng.make_axes([
        X,
        Y,
        C
    ])

    x_val = np.absolute(np.random.randn(*axes.lengths))
    y_val = np.absolute(np.random.randn(*bcast_axes.lengths))
    z_val = np.absolute(np.random.randn(*bcast_axes.lengths))

    x = ng.constant(x_val, axes)
    y = ng.constant(y_val, bcast_axes)
    z = ng.constant(z_val, bcast_axes)

    out = ng.add(ng.add(x, y), z)

    with executor(out) as ex:
        graph_val = ex()
    np_val = np.add(np.add(x_val.reshape(32, 32, 1), y_val), z_val)
    ng.testing.assert_allclose(graph_val, np_val, rtol=1e-4)
