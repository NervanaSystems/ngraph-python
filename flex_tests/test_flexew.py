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
from ngraph.testing import executor, assert_allclose
from ngraph.testing import ExecutorFactory

pytestmark = pytest.mark.transformer_dependent("module")


def test_abs(transformer_factory):
    """
    absolute value of matrix
    """
    n, m = 2, 3
    N = ng.make_axis(length=n)
    M = ng.make_axis(length=m)
    Zin = ng.placeholder((N, M))
    Zout = abs(Zin)

    with executor(Zout, Zin) as ex:
        abs_executor = ex

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

    with executor(y,x) as ex:
        sum_executor = ex

        xval = np.array(list(range(1,10)) + [-1])
        xval[2] += + 1 + 10

        assert(sum_executor(xval) == sum(xval))


def test_plusconst(transformer_factory):
    """
    x + 1.5
    """
    x = ng.placeholder(())
    x_plus_const = x + 1.5

    with executor(x_plus_const, x) as ex:
        plusconst_executor = ex

        for i in range(5):
            # 8.8 fixed point test
            # assert plusconst_executor(i) == i + 1.5

            # autoflex test
            if i == 1:
                # expect overflow
                assert_allclose(plusconst_executor(i), 1.9999)
            else:
                assert plusconst_executor(i) == i + 1.5
