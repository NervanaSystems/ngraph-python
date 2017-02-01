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
from ngraph.testing import ExecutorFactory


def test_variable_init(transformer_factory):
    C = ng.make_axis().named('C')
    C.length = 200

    w_init = np.random.rand(C.length)
    W = ng.variable(ng.make_axes([C]), initial_value=w_init)

    with ExecutorFactory() as ex:
        result = ex.executor(W)()
    ng.testing.assert_allclose(result, w_init)


def test_deriv_missing_connection():
    """
    Taking the derivative of an expression with respect to a variable not
    used to compute the expression should raise an exception.
    """

    N = ng.make_axis(1)

    x = ng.variable([N])
    y = ng.variable([N])
    z = ng.variable([N])

    with pytest.raises(ValueError):
        ng.deriv(x + y, z)


def test_sequential():
    N = ng.make_axis(1)
    x = ng.variable([N], initial_value=0)
    with ng.sequential_op_factory() as pf:
        x0 = x + x
        ng.assign(x, 2)
        x1 = x + x
        pf.append(x0)
    p = pf()
    with ExecutorFactory() as ex:
        x0_val, x1_val, p_val = ex.executor([x0, x1, p])()
    assert x0_val == 0
    assert x1_val == 4
    assert p_val == 0


def test_sequential_reduce():
    N = ng.make_axis(3)
    x = ng.variable([N], initial_value=1)
    with ng.sequential_op_factory() as pf:
        x0 = x + x
        x1 = ng.sum(x0, out_axes=())
        x2 = ng.sum(x0, out_axes=()) + x0
    p = pf()
    with ExecutorFactory() as ex:
        x0_val, x1_val, x2_val, p_val, x_val = ex.executor([x0, x1, x2, p, x])()
        x0_np = x_val + x_val
        x1_np = np.sum(x0_np)
        x2_np = x1_np + x0_np
        assert np.allclose(x0_val, x0_np)
        assert np.allclose(x1_val, x1_np)
        assert np.allclose(x2_val, x2_np)
        assert np.allclose(p_val, x2_np)


@pytest.mark.skip(reason="Need value_op to correctly check side-effects")
def test_sequential_side():
    N = ng.make_axis(3)
    x = ng.variable([N], initial_value=[1, 2, 3])
    x1 = ng.persistent_tensor(axes=(), initial_value=2)
    x2 = ng.persistent_tensor(axes=(), initial_value=3)
    b = ng.persistent_tensor(axes=(), initial_value=1)

    with ng.sequential_op_factory() as pf:
        ng.assign(x1, ng.sum(x, out_axes=()) + x1 * b + (1 - b))
        ng.assign(x2, ng.mean(x, out_axes=()) + x2 * b + (1 - b))
        y = x * 2
        pf.append(y)
    p = pf()

    with ExecutorFactory() as ex:
        main_effect = ex.executor(p)

    # Run main path #1
    y_val = main_effect()
    x_np = np.array([1, 2, 3], dtype=np.float32)
    y_np = x_np * 2

    assert np.allclose(y_val, y_np)

    # Run main path #2 (Should be the same as before)
    y_val = main_effect()

    assert np.allclose(y_val, y_np)

    # TODO: use value_op for this type of retrieval instead
    # Now check side effects
    x1_val, x2_val = x1.value.tensor, x2.value.tensor

    x1_np = x_np.sum() + (x_np.sum() + 2)
    x2_np = x_np.mean() + (x_np.mean() + 3)

    assert np.allclose(x2_val, x2_np)
    assert np.allclose(x1_val, x1_np)


def test_pad_invalid_paddings_length():
    """
    pad should raise an exception if the paddings length is not the same as the
    input dimensionality.
    """
    N = ng.make_axis(1)

    x = ng.variable([N])
    with pytest.raises(ValueError):
        ng.pad(x, [1, 0])


def test_pad_0():
    """
    pad with length 0 should be a nop
    """

    N = ng.make_axis(1)

    x = ng.variable([N])

    assert ng.pad(x, [0]).axes == x.axes


def test_pad_mixed():
    """
    mix 0 padding with non-0 padding
    """

    N = ng.make_axis(1)
    M = ng.make_axis(1)

    x = ng.variable([N, M])

    pad = ng.pad(x, [0, 1])

    assert pad.axes[0] == x.axes[0]
    assert pad.axes[1] != x.axes[1]


def test_slice_nop():
    """
    slicing with nop slice should return same axis
    """

    N = ng.make_axis(1)
    M = ng.make_axis(1)

    x = ng.variable([N, M])

    s = ng.tensor_slice(x, [
        slice(None, None, None),
        slice(None, None, -1),
    ])

    assert s.axes[0] == x.axes[0]
    assert s.axes[1] != x.axes[1]


def test_tensor_slice():
    """
    slicing a tensor should work like numpy
    """

    M = ng.make_axis(10)
    N = ng.make_axis(20)
    O = ng.make_axis(5)

    x = ng.placeholder(axes=[M, N, O])

    assert x[:5].axes.full_lengths == (5, 20, 5)
    assert x[:, 2:7].axes.full_lengths == (10, 5, 5)
    assert x[:5, :, :-1].axes.full_lengths == (5, 20, 4)


def test_setting():
    with ExecutorFactory() as ex:
        X = ng.make_axis(length=3).named('X')
        axes = ng.make_axes([X])

        np_x = np.array([1, 2, 3], dtype=np.float32)
        np_y = np.array([1, 3, 5], dtype=np.float32)

        x = ng.constant(np_x, axes)
        y = ng.constant(np_y, axes)

        v = ng.variable(axes, initial_value=x)

        f_v = ex.executor(v)

        vset = ng.sequential([
            ng.assign(v, v + y),
            v
        ])
        f_v1 = ex.executor(vset)

        f_v2 = ex.executor(v)

        e_v = f_v().copy()
        assert ng.testing.allclose(e_v, np_x)
        e_v1 = f_v1().copy()
        assert ng.testing.allclose(e_v1, np_x + np_y)
        e_v2 = f_v2().copy()
        assert ng.testing.allclose(e_v2, np_x + np_y)
