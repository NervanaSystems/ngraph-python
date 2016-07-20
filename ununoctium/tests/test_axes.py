from builtins import range
import numpy as np

import geon.backends.graph.arrayaxes as arrax
import geon.backends.graph.names as names

# Make some axes
ax = names.NameScope()

ax.A = arrax.Axis(10)
ax.B = arrax.Axis(15)
ax.C = arrax.Axis(20)
ax.D = arrax.Axis(25)


def to_nested_tup(axes):
    return tuple(to_nested_tup(axis.axes) if
                 isinstance(axis, arrax.AxesAxis) else axis for axis in axes)


def test_canonicalize_axes():
    def test(l, r):
        assert to_nested_tup(arrax.Axes(*l)) == r,\
            ('Failed. Original collection: %s, axes: %s,' +
             ' generated tuple: %s, target: %s') %\
            (l, arrax.Axes(*l), to_nested_tup(arrax.Axes(*l)), r)

    test((), ())
    test([], ())
    test([ax.A], (ax.A,))
    test([ax.A, (ax.B,)], (ax.A, ax.B))
    test([ax.A, (ax.B, ax.C)], (ax.A, (ax.B, ax.C)))


def test_axes_ops():
    # Subtraction
    def test_sub(axes1, axes2, target):
        assert arrax.AxisIDTuple.sub(
            axes1.as_axis_ids(),
            axes2.as_axis_ids()).as_axes() == target
    test_sub(arrax.Axes(ax.A, ax.B), arrax.Axes(ax.A,), arrax.Axes(ax.B,))
    test_sub(arrax.Axes(ax.A, ax.B), arrax.Axes(ax.B,), arrax.Axes(ax.A,))

    # Combined axes length
    assert arrax.AxesAxis(arrax.Axes(ax.A, ax.B,)).length \
        == ax.A.length * ax.B.length
    assert arrax.Axes(ax.A, (ax.B, ax.C)).lengths \
        == (ax.A.length, ax.B.length * ax.C.length)
    assert arrax.AxesAxis(arrax.Axes(ax.A, (ax.B, ax.C))).length \
        == ax.A.length * ax.B.length * ax.C.length


def empty(td):
    return np.empty(td.shape, td.dtype)


def tensorview(td, nparr):
    return np.ndarray(shape=td.shape, dtype=td.dtype,
                      buffer=nparr, offset=td.offset,
                      strides=td.strides)

# A scalar
td0 = arrax.TensorDescription(axes=())
e0 = empty(td0)

# A simple vector
td1 = arrax.TensorDescription(axes=[ax.A])
e1 = empty(td1)

td2 = arrax.TensorDescription(axes=[ax.A, ax.B])
e2 = empty(td2)

td3 = arrax.TensorDescription(axes=(ax.D, ax.D))
e3 = empty(td3)

# Reaxes
td0_1 = td0.reaxe([ax.A])
e0_1 = tensorview(td0_1, e0)

td0_2 = td0.reaxe([ax.A, ax.B])
e0_2 = tensorview(td0_2, e0)

td1_1 = td1.reaxe([ax.A, ax.B])
td1_2 = td1.reaxe([ax.B, ax.A])
td1_3 = td1.reaxe([(ax.B, ax.C), ax.A])
e1_1 = tensorview(td1_1, e1)
e1_2 = tensorview(td1_2, e1)
e1_3 = tensorview(td1_3, e1)

td2_1 = td2.reaxe(arrax.Axes(ax.B, ax.A))
e2_1 = tensorview(td2_1, e2)
td2_2 = td2.reaxe(arrax.Axes(ax.A, ax.B))
e2_2 = tensorview(td2_2, e2)
td2_3 = td2.reaxe(arrax.Axes(arrax.AxesAxis(td2.axes)))
e2_3 = tensorview(td2_3, e2)

td3_1 = td3.reaxe_with_axis_ids(arrax.AxisIDTuple(
    arrax.AxisID(ax.D, 1),
    arrax.AxisID(ax.D, 0)
))
e3_1 = tensorview(td3_1, e3)


def test_simple_tensors():
    val1 = 3

    e0[()] = val1
    assert all([e0_1[i] == e0 for i in range(ax.A.length)])

    for i in range(ax.A.length):
        for j in range(ax.B.length):
            assert e0_2[i, j] == e0

    assert e1_1.shape == (ax.A.length, ax.B.length)
    assert e1_2.shape == (ax.B.length, ax.A.length)

    for i in range(ax.A.length):
        e1_1[i] = i

    for i in range(ax.A.length):
        assert e1[i] == i
        for j in range(ax.B.length):
            assert e1_1[i, j] == i
            assert e1_2[j, i] == i
        for j in range(ax.B.length * ax.C.length):
            assert e1_3[j, i] == i

    def val2(i, j):
        return (i + 1) * (j + 2)

    for i in range(ax.A.length):
        for j in range(ax.B.length):
            e2[i, j] = val2(i, j)

    for i in range(ax.A.length):
        for j in range(ax.B.length):
            assert e2_1[j, i] == val2(i, j)
            assert e2_2[i, j] == val2(i, j)
            assert e2_3[i * ax.B.length + j] == val2(i, j)

    for i in range(ax.D.length):
        for j in range(ax.D.length):
            e3[i, j] = val2(i, j)

    for i in range(ax.D.length):
        for j in range(ax.D.length):
            assert e3[i, j] == e3_1[j, i]

if __name__ == '__main__':
    test_canonicalize_axes()
    test_axes_ops()
    test_simple_tensors()