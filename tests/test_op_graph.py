import pytest

import numpy as np
import ngraph as ng
from ngraph.util.utils import ExecutorFactory


def test_variable_init(transformer_factory):
    C = ng.make_axis("C")
    C.length = 200

    w_init = np.random.rand(C.length)
    W = ng.Variable(axes=ng.make_axes([C]), initial_value=w_init)

    ex = ExecutorFactory()
    result = ex.executor(W)()
    np.testing.assert_allclose(result, w_init)


def test_deriv_missing_connection():
    """
    Taking the derivative of an expression with respect to a variable not
    used to compute the expression should raise an exception.
    """

    N = ng.make_axis(1)

    x = ng.Variable(axes=[N])
    y = ng.Variable(axes=[N])
    z = ng.Variable(axes=[N])

    with pytest.raises(ValueError):
        ng.deriv(x + y, z)


def test_pad_invalid_paddings_length():
    """
    pad should raise an exception if the paddings length is not the same as the
    input dimensionality.
    """
    N = ng.make_axis(1)

    x = ng.Variable(axes=[N])
    with pytest.raises(ValueError):
        ng.pad(x, [1, 0])


def test_pad_0():
    """
    pad with length 0 should be a nop
    """

    N = ng.make_axis(1)

    x = ng.Variable(axes=[N])

    assert ng.pad(x, [0]).axes == x.axes


def test_pad_mixed():
    """
    mix 0 padding with non-0 padding
    """

    N = ng.make_axis(1)
    M = ng.make_axis(1)

    x = ng.Variable(axes=[N, M])

    pad = ng.pad(x, [0, 1])

    assert pad.axes[0] == x.axes[0]
    assert pad.axes[1] != x.axes[1]


def test_slice_nop():
    """
    slicing with nop slice should return same axis
    """

    N = ng.make_axis(1)
    M = ng.make_axis(1)

    x = ng.Variable(axes=[N, M])

    s = ng.Slice(x, [
        slice(None, None, None),
        slice(None, None, -1),
    ])

    assert s.axes[0] == x.axes[0]
    assert s.axes[1] != x.axes[1]


def test_setting():
    ex = ExecutorFactory()
    X = ng.make_axis(name='X', length=3)
    axes = ng.make_axes([X])

    np_x = np.array([1, 2, 3], dtype=np.float32)
    np_y = np.array([1, 3, 5], dtype=np.float32)

    x = ng.Constant(np_x, axes=axes)
    y = ng.Constant(np_y, axes=axes)

    v = ng.Variable(axes=axes, initial_value=x)

    f_v = ex.executor(v)

    with ng.Op.saved_user_deps():
        ng.assign(v, v + y)
        f_v1 = ex.executor(v)

    f_v2 = ex.executor(v)

    e_v = f_v().copy()
    assert np.allclose(e_v, np_x)
    e_v1 = f_v1().copy()
    assert np.allclose(e_v1, np_x + np_y)
    e_v2 = f_v2().copy()
    assert np.allclose(e_v2, np_x + np_y)
