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
Test the usage of ng.Constant
"""
from __future__ import print_function

import ngraph as ng
import ngraph.frontends.base.axis as ax
from ngraph.util.utils import executor
import numpy as np


def test_constant_init():
    """TODO."""
    a = ng.Constant(5)
    result = executor(a)()
    print(result)

    assert (result == 5)
    print("pass constant initialization")


def test_constant_add():
    """TODO."""
    a = ng.Constant(1)
    b = ng.Constant(2)
    c = a + b

    result = executor(c)()
    print(result)
    assert result == 3


def test_constant_multiply():
    """TODO."""
    a = ng.Constant(4)
    b = ng.Constant(2)
    c = ng.multiply(a, b)
    result = executor(c)()
    assert result == 8


def test_numpytensor_add():
    """TODO."""
    ax.Y.length = 2
    a = ng.Constant(np.array([3, 5], dtype=np.float32), axes=[ax.Y])
    b = ng.Constant(np.array([3, 5], dtype=np.float32), axes=[ax.Y])
    c = a + b
    result = executor(c)()
    assert np.array_equal(result, [6, 10])

    np_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    np_b = np.array([[1, 2], [3, 4]], dtype=np.float32)
    np_c = np_a + np_b

    ax.M.length = 2
    ax.N.length = 2
    a = ng.Constant(np_a, axes=[ax.M, ax.N])
    b = ng.Constant(np_b, axes=[ax.M, ax.N])
    c = a + b
    result = executor(c)()
    assert np.array_equal(result, np_c)


def test_numpytensor_dot():
    np_a = np.array([[1, 2, 3]], dtype=np.float32)
    np_b = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.float32)
    np_c = np.dot(np_a, np_b)

    ax.M.length = 1
    ax.N.length = 3
    a = ng.Constant(np_a, axes=[ax.M, ax.N])
    ax.N.length = 3
    ax.Y.length = 2
    b = ng.Constant(np_b, axes=[ax.N, ax.Y])
    c = ng.dot(a, b)
    result = executor(c)()

    assert np.array_equal(result, np_c)


def test_numpytensor_multiply_constant():
    """TODO."""
    np_a = np.array([[1, 2, 3]], dtype=np.float32)
    np_c = np.multiply(np_a, 2)

    ax.M.length = 1
    ax.N.length = 3
    a = ng.Constant(np_a, axes=[ax.M, ax.N])
    b = ng.Constant(2)
    c = ng.multiply(a, b)
    result = executor(c)()
    print(result)
    assert np.array_equal(result, np_c)


def test_numpytensor_add_constant():
    """TODO."""
    np_a = np.array([[1, 2, 3]], dtype=np.float32)
    np_c = np.add(np_a, 2)

    ax.M.length = 1
    ax.N.length = 3
    a = ng.Constant(np_a, axes=[ax.M, ax.N])
    b = ng.Constant(2)
    c = ng.add(a, b)
    result = executor(c)()
    print(result)
    assert np.array_equal(result, np_c)


def test_fusion():
    """TODO."""
    np_a = np.array([[1, 2, 3]], dtype=np.float32)
    np_b = np.array([[3, 2, 1]], dtype=np.float32)
    np_d = np.multiply(np_b, np.add(np_a, 2))

    ax.M.length = 1
    ax.N.length = 3
    a = ng.Constant(np_a, axes=[ax.M, ax.N])
    b = ng.Constant(np_b, axes=[ax.M, ax.N])
    c = ng.Constant(2)
    d = ng.multiply(b, ng.add(a, c))
    result = executor(d)()
    print(result)
    assert np.array_equal(result, np_d)


def test_numpytensor_mlp():
    """TODO."""
    np_x = np.array([[1, 2, 3]], dtype=np.float32)
    np_w = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.float32)
    np_b = np.array([1, 2], dtype=np.float32)
    np_c = np.dot(np_x, np_w) + np_b

    ax.N.length = 1
    ax.D.length = 3
    ax.H.length = 2
    x = ng.Constant(np_x, axes=[ax.N, ax.D])
    w = ng.Constant(np_w, axes=[ax.D, ax.H])
    b = ng.Constant(np_b, axes=[ax.H])
    wx = ng.dot(x, w)
    c = wx + b
    result = executor(c)()
    print(result)
    print(np_c)
    assert np.array_equal(result, np_c)
