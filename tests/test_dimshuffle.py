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
import pytest
import numpy as np

import ngraph as ng
from ngraph.util.derivative_check import check_derivative
from ngraph.util.utils import executor
from ngraph.util.utils import RandomTensorGenerator


rng = RandomTensorGenerator(0, np.float32)


def test_dimshuffle_fprop(transformer_factory):
    """
    dimshuffle a 2d array and make sure fprop works
    """
    A = ng.Axis(2)
    B = ng.Axis(3)

    x = ng.placeholder(axes=ng.Axes([A, B]))

    # compute convolution with graph
    output = ng.Dimshuffle(x, axes=ng.Axes([B, A]))

    assert output.axes == ng.Axes([B, A])

    # randomly initialize
    x_value = rng.uniform(-1, 1, x.axes)

    result = executor(output, x)(x_value)

    np.testing.assert_allclose(result, x_value.T)


def test_dimshuffle_bprop(transformer_factory):
    """
    dimshuffle a 2d array and make sure bprop works
    """
    A = ng.Axis(2)
    B = ng.Axis(3)

    x = ng.placeholder(axes=ng.Axes([A, B]))

    # randomly initialize
    x_value = rng.uniform(-1, 1, x.axes)

    check_derivative(
        ng.Dimshuffle(x, axes=ng.Axes([B, A])),
        x, 0.001, x_value,
        atol=1e-3, rtol=1e-3
    )


@pytest.fixture
def A():
    return ng.Axis(2)


@pytest.fixture
def B():
    return ng.Axis(3)


@pytest.fixture
def C():
    return ng.Axis(4)


@pytest.fixture
def x(A, B, C):
    return ng.placeholder(axes=ng.Axes([A, B]))


def test_fail_on_missing(transformer_factory, x, B):
    with pytest.raises(ValueError):
        ng.Dimshuffle(x, axes=ng.Axes([B, B]))


def test_fail_on_extra_axis(transformer_factory, x, A, B, C):
    with pytest.raises(ValueError):
        ng.Dimshuffle(x, axes=ng.Axes([A, B, C]))


def test_fail_on_missing_and_extra_axis(transformer_factory, x, A, C):
    with pytest.raises(ValueError):
        ng.Dimshuffle(x, axes=ng.Axes([A, C]))


def test_fail_on_axis_reuse(transformer_factory, x, A, B):
    with pytest.raises(ValueError):
        ng.Dimshuffle(x, axes=ng.Axes([A, B, B]))
