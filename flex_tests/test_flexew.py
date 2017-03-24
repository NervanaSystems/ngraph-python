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


# def test_abs(transformer_factory):
#     """
#     absolute value of matrix
#     """
#     n, m = 2, 3
#     N = ng.make_axis(length=n)
#     M = ng.make_axis(length=m)
#     Zin = ng.placeholder((N, M))
#     Zout = abs(Zin)
#
#     with executor(Zout, Zin) as ex:
#         abs_executor = ex
#
#         Xval = np.array([5, 1, 0, -2, 3, 4]).reshape(n, m).astype(np.float32)
#         Xval[0,1] = -Xval[0,1]
#         assert np.allclose(abs_executor(Xval), abs(Xval))
#
#
# def test_sum(transformer_factory):
#     """
#     sum 1-D tensor
#     """
#
#     nelems = 10
#     H = ng.make_axis(length=nelems)
#     x = ng.placeholder(H)
#     y = ng.sum(x)
#
#     with executor(y,x) as ex:
#         sum_executor = ex
#
#         xval = np.array(list(range(1,10)) + [-1])
#         xval[2] += + 1 + 10
#
#         assert(sum_executor(xval) == sum(xval))
#
#
# def test_plusconst(transformer_factory):
#     """
#     x + 1.5
#     """
#     x = ng.placeholder(())
#     x_plus_const = x + 1.5
#
#     with executor(x_plus_const, x) as ex:
#         plusconst_executor = ex
#
#         for i in range(5):
#             # 8.8 fixed point test
#             # assert plusconst_executor(i) == i + 1.5
#
#             # autoflex test
#             if i == 1:
#                 # expect overflow
#                 assert_allclose(plusconst_executor(i), 1.9999)
#             else:
#                 assert plusconst_executor(i) == i + 1.5

def test_assign(transformer_factory):
    # x = ng.placeholder((), initial_value=0)
    M = ng.make_axis(length=0)
    axes = ng.make_axes([M])
    x = ng.variable(axes, initial_value=0)
    ng_fun = ng.assign(x, 5)
    print x

    with executor() as ex:
        flex = ex()
        print flex



def test_setting():
    with ExecutorFactory() as ex:
        # M = ng.make_axis(length=1)
        # axes = ng.make_axes([M])
        v = ng.variable(())
        vset2 = ng.sequential([
            ng.assign(v, 0.4),
            v
        ])
        f_v12 = ex.executor(vset2)
        e_v12 = f_v12()
        print e_v12


def test_setting2(transformer_factory):
    # # M = ng.make_axis(length=1)
    # # axes = ng.make_axes([M])
    # v = ng.variable(())
    # vset2 = ng.sequential([
    #     ng.assign(v, 0.4),
    #     v
    # ])
    # with executor(vset2) as ex:
    #     e_v12 = ex()
    #     print e_v12
    x = ng.placeholder(())
    z = ng.value_of(x)
    with executor(z, x) as ex:
        print ex(0.4)

# def test_assign_2(transformer_factory):
#     M = ng.make_axis(length=0)
#     axes = ng.make_axes([M])
#     x = ng.variable(axes, initial_value=0)
#     x0 = x + x
#     x1 = x + x
#     p = ng.sequential([
#         x0,
#         ng.assign(x, 2),
#         x1,
#         x0
#     ])
#     with executor() as ex:
#         x0_val, x1_val, p_val = ex.executor([x0, x1, p])()
#     assert x0_val == 0
#     assert x1_val == 4
#     assert p_val == 0
