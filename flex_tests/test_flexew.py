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
from ngraph.testing import executor


def test_abs(transformer_factory):
    """
    absolute value of matrix
    """
    n, m = 2, 3
    N = ng.make_axis(length=n)
    M = ng.make_axis(length=m)
    Zin = ng.placeholder((N, M))
    Zout = abs(Zin)

    abs_executor = executor(Zout, Zin)

    Xval = np.array([5, 1, 0, -2, 3, 4]).reshape(n, m).astype(np.float32)
    Xval[0,1] = -Xval[0,1]
    assert np.allclose(abs_executor(Xval), abs(Xval))


def test_sum(transformer_factory):
    """
    sum 1-D tensor
    """

    nelems = 10
    H = ng.make_axis(length=nelems)
    x = ng.placeholder(H)
    y = ng.sum(x)

    sum_executor = executor(y, x)

    xval = np.array(list(range(1,10)) + [-1])
    xval[2] += + 1 + 10

    assert(sum_executor(xval) == sum(xval))


def test_plusconst(transformer_factory):
    """
    x + 1.5
    """
    x = ng.placeholder(())
    x_plus_const = x + 1.5

    plusconst_executor = executor(x_plus_const, x)

    for i in range(5):
        assert plusconst_executor(i) == i + 1.5
