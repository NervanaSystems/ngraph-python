# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
import ngraph as ng
from ngraph.testing import ExecutorFactory


def test_read_state():
    """
    This just reads back a tensor. No code is generated.
    """
    with ExecutorFactory() as ex:
        N = ng.make_axis(3, name='N')
        x_np = np.ones((N.length)) * 4
        x = ng.variable([N], initial_value=x_np).named('x')
        f = ex.executor(x)
        x_val = f()
        assert np.allclose(x_np, x_val)


def test_write_state():
    """
    This reads back a tensor set from an argument. No code is generated.
    """
    with ExecutorFactory() as ex:
        N = ng.make_axis(3, name='N')
        x_np = np.ones((N.length)) * 4
        x = ng.persistent_tensor([N]).named('x')
        f = ex.executor(x, x)
        x_val = f(x_np)
        assert np.allclose(x_np, x_val)


def test_use_state():
    """
    Uses the value of a tensor in a computation.
    """
    with ExecutorFactory() as ex:
        N = ng.make_axis(3, name='N')
        x_np = np.ones((N.length)) * 4
        x = ng.variable([N], initial_value=x_np).named('x')
        xx = x + x
        f = ex.executor(xx)
        xx_val = f()
        assert np.allclose(x_np + x_np, xx_val)


def test_modify_state():
    with ExecutorFactory() as ex:
        N = ng.make_axis(3, name='N')
        x_np = np.ones((N.length)) * 4
        x = ng.variable([N], initial_value=x_np).named('x')
        val = ng.sequential([
            ng.assign(x, x + x),
            x
        ])
        f = ex.executor(val)
        x_val = f()
        assert np.allclose(x_np + x_np, x_val)


def test_fill_state():
    with ExecutorFactory() as ex:
        N = ng.make_axis(3, name='N')
        x_np = np.ones((N.length)) * 4
        x = ng.variable([N], initial_value=x_np).named('x')
        val = ng.sequential([
            ng.fill(x, -1),
            x
        ])
        f = ex.executor(val)
        x_val = f()
        assert np.allclose(-1, x_val)


def test_concatenate():
    with ExecutorFactory() as ex:
        A = ng.make_axis(name='A', length=3)
        B = ng.make_axis(name='B', length=4)
        np_shape = (A.length, B.length)
        x0_np = -np.ones(np_shape)
        x1_np = np.ones(np_shape)
        x0_ng = ng.persistent_tensor([A, B], initial_value=x0_np).named('x0')
        x1_ng = ng.persistent_tensor([A, B], initial_value=x1_np).named('x1')
        j_np = np.concatenate([x0_np, x1_np], axis=0)
        j_ng = ng.concat_along_axis([x0_ng, x1_ng], A)
        f = ex.executor(j_ng)
        j_val = f()
        assert ng.testing.allclose(j_val, j_np)


def test_specific_slice_deriv():
    #
    with ExecutorFactory() as ex:
        A = ng.make_axis(name='A', length=3)
        B = ng.make_axis(name='B', length=4)
        np_shape = (A.length, B.length)
        x_np = np.empty(np_shape, dtype=np.float32)
        for i in range(A.length):
            for j in range(B.length):
                x_np[i, j] = 10 * i + j
        x_ng = ng.persistent_tensor([A, B], initial_value=x_np)
        for i in range(A.length):
            for j in range(B.length):
                slice = ng.tensor_slice(x_ng, (i, j))
                dslice_dx = ng.deriv(slice, x_ng)
                dslice_dx_fun = ex.executor(dslice_dx)
                dslice_dx_val = dslice_dx_fun()
                dslice_dx_np = np.zeros_like(x_np)
                dslice_dx_np[i, j] = 1
                assert ng.testing.allclose(dslice_dx_val, dslice_dx_np)


def test_slice_deriv():
    C = ng.make_axis(length=2)
    D = ng.make_axis(length=3)

    x_np = np.array([[10, 20, 30], [1, 2, 3]], dtype='float32')
    x = ng.placeholder([C, D]).named('x')

    x_slice = x[0, :] + x[1, :]

    with ExecutorFactory() as ex:
        sym_deriv_fun = ex.derivative(x_slice, x)
        val_ng = sym_deriv_fun(x_np)
        val_np = np.zeros((D.length, C.length, D.length))
        for i in range(D.length):
            for j in range(C.length):
                val_np[i, j, i] = 1
        assert ng.testing.allclose(val_ng, val_np)
