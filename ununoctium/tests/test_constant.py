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
'''
Test the usage of be.Constant

'''
from __future__ import print_function

import geon.backends.graph.funs as be
import geon.backends.graph.axis as ax
import numpy as np


def test_constant_init():
    with be.bound_environment():
        a = be.Constant(5)

        enp = be.NumPyTransformer(results=[a])

        result = enp.evaluate()[a]
        print(result)

        assert (result == 5)
        print("pass constant initialization")


def test_constant_add():
    with be.bound_environment():
        a = be.Constant(1)
        b = be.Constant(2)
        c = a + b

        enp = be.NumPyTransformer(results=[c])
        result = enp.evaluate()[c]
        print(result)
        assert result == 3


def test_constant_multiply():
    with be.bound_environment():
        a = be.Constant(4)
        b = be.Constant(2)
        c = be.multiply(a, b)
        enp = be.NumPyTransformer(results=[c])
        result = enp.evaluate()[c]
        assert result == 8


def test_numpytensor_add():
    with be.bound_environment():
        ax.Y.length = 2
        a = be.NumPyTensor(np.array([3, 5], dtype=np.float32), axes=[ax.Y])
        b = be.NumPyTensor(np.array([3, 5], dtype=np.float32), axes=[ax.Y])
        c = a + b
        enp = be.NumPyTransformer(results=[c])
        result = enp.evaluate()
        assert np.array_equal(result[c], [6, 10])

        np_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        np_b = np.array([[1, 2], [3, 4]], dtype=np.float32)
        np_c = np_a + np_b

        ax.M.length = 2
        ax.N.length = 2
        a = be.NumPyTensor(np_a, axes=[ax.M, ax.N])
        b = be.NumPyTensor(np_b, axes=[ax.M, ax.N])
        c = a + b
        enp = be.NumPyTransformer(results=[c])
        result = enp.evaluate()
        assert np.array_equal(result[c], np_c)


def test_numpytensor_dot():
    with be.bound_environment():
        np_a = np.array([[1, 2, 3]], dtype=np.float32)
        np_b = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.float32)
        np_c = np.dot(np_a, np_b)

        ax.M.length = 1
        ax.N.length = 3
        a = be.NumPyTensor(np_a, axes=[ax.M, ax.N])
        ax.N.length = 3
        ax.Y.length = 2
        b = be.NumPyTensor(np_b, axes=[ax.N, ax.Y])
        c = be.dot(a, b)
        enp = be.NumPyTransformer(results=[c])
        result = enp.evaluate()

        assert np.array_equal(result[c], np_c)


test_constant_init()
test_constant_add()
test_constant_multiply()
test_numpytensor_add()
test_numpytensor_dot()
