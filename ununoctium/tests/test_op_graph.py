import pytest

import geon as be


def test_deriv_missing_connection():
    """
    Taking the derivative of an expression with respect to a variable not
    used to compute the expression should raise an exception.
    """

    N = be.Axis(1)

    x = be.Variable(axes=[N])
    y = be.Variable(axes=[N])
    z = be.Variable(axes=[N])

    with pytest.raises(ValueError):
        be.deriv(x + y, z)


def test_pad_invalid_paddings_length():
    """
    pad should raise an exception if the paddings length is not the same as the
    input dimensionality.
    """
    N = be.Axis(1)

    x = be.Variable(axes=[N])
    with pytest.raises(ValueError):
        be.pad(x, [1, 0])


def test_pad_0():
    """
    pad with length 0 should be a nop
    """

    N = be.Axis(1)

    x = be.Variable(axes=[N])

    assert be.pad(x, [0]).axes == x.axes


def test_pad_mixed():
    """
    mix 0 padding with non-0 padding
    """

    N = be.Axis(1)
    M = be.Axis(1)

    x = be.Variable(axes=[N, M])

    pad = be.pad(x, [0, 1])

    assert pad.axes[0] == x.axes[0]
    assert pad.axes[1] != x.axes[1]


def test_slice_nop():
    """
    slicing with nop slice should return same axis
    """

    N = be.Axis(1)
    M = be.Axis(1)

    x = be.Variable(axes=[N, M])

    s = be.Slice(x, [
        slice(None, None, None),
        slice(None, None, -1),
    ])

    assert s.axes[0] == x.axes[0]
    assert s.axes[1] != x.axes[1]
