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
from __future__ import print_function
import numpy as np

from neon.initializers import Constant, Array

'''
Test the usage of be.Variable

'''
import geon.backends.graph.funs as be
import geon.backends.graph.axis as ax


def test_init_variable():
    """TODO."""
    with be.bound_environment():
        ax.Y.length = 1
        hello = be.Variable(axes=(ax.Y,), init=Constant(4))

        transformer = be.NumPyTransformer()
        comp = transformer.computation(hello)

        result = comp()
        print(result)

        assert (result == 4)
        print("pass variable initialization")


def test_init_1D_variable():
    """TODO."""
    with be.bound_environment():
        ax.Y.length = 2
        hello = be.Variable(axes=(ax.Y,), init=Constant(3))

        transformer = be.NumPyTransformer()
        comp = transformer.computation(hello)

        result = comp()
        print(result)

        print("pass 1D variable initialization")


def test_init_2D_variable():
    """TODO."""
    with be.bound_environment():
        ax.M.length = 3
        ax.N.length = 2
        hello = be.Variable(axes=[ax.M, ax.N], init=Constant(5))

        transformer = be.NumPyTransformer()
        comp = transformer.computation(hello)

        result = comp()[hello]
        print(result)

        print("pass 2D variable initialization")


def test_init_1D_variable_from_numpy_array():
    """TODO."""
    with be.bound_environment():
        ax.Y.length = 10
        npvar = be.NumPyTensor(
            np.arange(ax.Y.length, dtype=np.float32), axes=[ax.Y])
        hello = be.Variable(axes=[ax.Y], init=Array(npvar))

        transformer = be.NumPyTransformer()
        comp = transformer.computation(hello)

        result = comp()
        print(result)

        print("pass numpy variable initialization")


def test_assign_1D_variable_with_numpy_tensor():
    """TODO."""
    with be.bound_environment():
        ax.Y.length = 10
        hello = be.Variable(axes=[ax.Y])
        npvar = be.NumPyTensor(
            np.arange(ax.Y.length, dtype=np.float32), axes=[ax.Y])

        hello = npvar
        transformer = be.NumPyTransformer()
        comp = transformer.computation(hello)
        result = comp()
        print(result)

        print("pass 1D numpy tensor assignment")


def test_assign_2D_variable_with_numpy_tensor():
    """TODO."""
    with be.bound_environment():
        ax.M.length = 3
        ax.N.length = 2

        var = be.Variable(axes=[ax.M, ax.N])
        npvar = be.NumPyTensor(
            np.array([[1, 2], [3, 4], [5, 6]]), axes=[ax.M, ax.N])

        op = be.assign(var, npvar)

        transformer = be.NumPyTransformer()
        comp = transformer.computation(op)
        result = comp()
        print(result)
        print(var.value)
