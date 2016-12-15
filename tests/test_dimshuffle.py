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
import numpy as np
import pytest

import ngraph as ng
from ngraph.testing import check_derivative, RandomTensorGenerator, executor


rng = RandomTensorGenerator(0, np.float32)


def test_dimshuffle_fprop(transformer_factory):
    """
    dimshuffle a 2d array and make sure fprop works
    """
    A = ng.make_axis(2)
    B = ng.make_axis(3)

    x = ng.placeholder(ng.make_axes([A, B]))

    # compute convolution with graph
    output = ng.axes_with_order(x, [B, A])

    assert output.axes == ng.make_axes([B, A])

    # randomly initialize
    x_value = rng.uniform(-1, 1, x.axes)

    result = executor(output, x)(x_value)

    ng.testing.assert_allclose(result, x_value.T)


def test_dimshuffle_bprop(transformer_factory):
    """
    dimshuffle a 2d array and make sure bprop works
    """
    A = ng.make_axis(2)
    B = ng.make_axis(3)

    x = ng.placeholder(ng.make_axes([A, B]))

    # randomly initialize
    x_value = rng.uniform(-1, 1, x.axes)

    check_derivative(
        ng.axes_with_order(x, [B, A]),
        x, 0.001, x_value,
        atol=1e-3, rtol=1e-3
    )


@pytest.fixture
def A():
    return ng.make_axis(2)


@pytest.fixture
def B():
    return ng.make_axis(3)


@pytest.fixture
def C():
    return ng.make_axis(4)


@pytest.fixture
def x(A, B, C):
    return ng.placeholder(ng.make_axes([A, B]))


def test_fail_on_missing(transformer_factory, x, B):
    with pytest.raises(ValueError):
        ng.axes_with_order(x, [B, B])


def test_fail_on_extra_axis(transformer_factory, x, A, B, C):
    with pytest.raises(ValueError):
        ng.axes_with_order(x, [A, B, C])


def test_fail_on_missing_and_extra_axis(transformer_factory, x, A, C):
    with pytest.raises(ValueError):
        ng.axes_with_order(x, [A, C])


def test_fail_on_axis_reuse(transformer_factory, x, A, B):
    with pytest.raises(ValueError):
        ng.axes_with_order(x, [A, B, B])
