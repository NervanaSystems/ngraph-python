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
Test the usage of ng.constant
"""
from __future__ import print_function

import numpy as np

import ngraph as ng
from ngraph.testing import executor
import pytest

pytestmark = [pytest.mark.transformer_dependent, pytest.mark.separate_execution]


def test_constant_init(transformer_factory):
    """TODO."""
    a = ng.constant(5)
    with executor(a) as ex:
        result = ex()
    print(result)

    assert (result == 5)
    print("pass constant initialization")

    nparray = np.array(range(5))
    a = ng.constant(nparray)
    with executor(a) as ex:
        result = ex()
    ng.testing.assert_allclose(result, nparray)


def test_constant_add(transformer_factory):
    """TODO."""
    a = ng.constant(1)
    b = ng.constant(2)
    c = a + b

    with executor(c) as ex:
        result = ex()
    print(result)
    assert result == 3


def test_constant_multiply(transformer_factory):
    """TODO."""
    a = ng.constant(4)
    b = ng.constant(2)
    c = ng.multiply(a, b)
    with executor(c) as ex:
        result = ex()
    assert result == 8


def test_cputensor_add(transformer_factory):
    """TODO."""
    Y = ng.make_axis(length=2)
    M = ng.make_axis(length=2)
    N = ng.make_axis(length=2)

    a = ng.constant(np.array([3, 5], dtype=np.float32), [Y])
    b = ng.constant(np.array([3, 5], dtype=np.float32), [Y])
    c = a + b
    with executor(c) as ex:
        result = ex()
    assert np.array_equal(result, [6, 10])

    np_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    np_b = np.array([[1, 2], [3, 4]], dtype=np.float32)
    np_c = np_a + np_b

    a = ng.constant(np_a, [M, N])
    b = ng.constant(np_b, [M, N])
    c = a + b
    with executor(c) as ex:
        result = ex()
    assert np.array_equal(result, np_c)


def test_cputensor_dot(transformer_factory):
    Y = ng.make_axis(length=2)
    M = ng.make_axis(length=1)
    N = ng.make_axis(length=3)

    np_a = np.array([[1, 2, 3]], dtype=np.float32)
    np_b = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.float32)
    np_c = np.dot(np_a, np_b)

    a = ng.constant(np_a, [M, N])
    b = ng.constant(np_b, [N, Y])
    c = ng.dot(a, b)

    with executor(c) as ex:
        result = ex()

    assert np.array_equal(result, np_c)


def test_cputensor_multiply_constant(transformer_factory):
    """TODO."""
    M = ng.make_axis(length=1)
    N = ng.make_axis(length=3)

    np_a = np.array([[1, 2, 3]], dtype=np.float32)
    np_c = np.multiply(np_a, 2)

    a = ng.constant(np_a, [M, N])
    b = ng.constant(2)
    c = ng.multiply(a, b)

    with executor(c) as ex:
        result = ex()
    print(result)
    assert np.array_equal(result, np_c)


def test_cputensor_add_constant(transformer_factory):
    """TODO."""
    M = ng.make_axis(length=1)
    N = ng.make_axis(length=3)

    np_a = np.array([[1, 2, 3]], dtype=np.float32)
    np_c = np.add(np_a, 2)

    a = ng.constant(np_a, [M, N])
    b = ng.constant(2)
    c = ng.add(a, b)
    with executor(c) as ex:
        result = ex()
    print(result)
    assert np.array_equal(result, np_c)


def test_cputensor_fusion(transformer_factory):
    """TODO."""
    M = ng.make_axis(length=1)
    N = ng.make_axis(length=3)

    np_a = np.array([[1, 2, 3]], dtype=np.float32)
    np_b = np.array([[3, 2, 1]], dtype=np.float32)
    np_d = np.multiply(np_b, np.add(np_a, 2))

    a = ng.constant(np_a, [M, N])
    b = ng.constant(np_b, [M, N])
    c = ng.constant(2)
    d = ng.multiply(b, ng.add(a, c))

    with executor(d) as ex:
        result = ex()
    print(result)
    assert np.array_equal(result, np_d)


def test_cputensor_mlp(transformer_factory):
    """TODO."""
    D = ng.make_axis(length=3)
    H = ng.make_axis(length=2)
    N = ng.make_axis(length=1)

    np_x = np.array([[1, 2, 3]], dtype=np.float32)
    np_w = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.float32)
    np_b = np.array([1, 2], dtype=np.float32)
    np_c = np.dot(np_x, np_w) + np_b

    x = ng.constant(np_x, [N, D])
    w = ng.constant(np_w, [D, H])
    b = ng.constant(np_b, [H])
    wx = ng.dot(x, w)
    c = wx + b
    with executor(c) as ex:
        result = ex()
    print(result)
    print(np_c)
    assert np.array_equal(result, np_c)
