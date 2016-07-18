import numpy as np

from geon.backends.graph.arrayaxes import Axis, axes_sub, canonicalize_axes, TensorDescription, axes_shape, axes_size
import geon.backends.graph.names as names

# Make some axes
ax = names.NameScope()

ax.A = Axis(10)
ax.B = Axis(15)
ax.C = Axis(20)
ax.D = Axis(25)


def test_canonicalize_axes():
    assert canonicalize_axes(()) == ()
    assert canonicalize_axes([]) == ()
    assert canonicalize_axes([ax.A]) == (ax.A,)
    assert canonicalize_axes([ax.A, (ax.B,)]) == (ax.A, ax.B)
    assert canonicalize_axes([ax.A, (ax.B, ax.C)]) == (ax.A, (ax.B, ax.C))


def test_axes_ops():
    assert canonicalize_axes(axes_sub([ax.A, ax.B], [ax.A])) == (ax.B,)
    assert canonicalize_axes(axes_sub([ax.A, ax.B], [ax.B])) == (ax.A,)

    assert list(axes_shape([ax.A, ax.B])) == [ax.A.length, ax.B.length]
    assert list(axes_shape([ax.A, [ax.B, ax.C]])) == [ax.A.length, ax.B.length * ax.C.length]

    assert axes_size([ax.A, ax.B]) == ax.A.length * ax.B.length
    assert axes_size([ax.A, [ax.B, ax.C]]) == ax.A.length * ax.B.length * ax.C.length


def empty(tensor_description):
    return np.empty(tensor_description.sizes, tensor_description.dtype)


def tensorview(tensor_description, buffer):
    return np.ndarray(shape=tensor_description.shape, dtype=tensor_description.dtype,
                      buffer=buffer,
                      offset=tensor_description.offset, strides=tensor_description.strides)


# A scalar
td0 = TensorDescription(axes=())

# A simple vector
td1 = TensorDescription(axes=[ax.A])

td2 = TensorDescription(axes=[ax.A, ax.B])


def test_strides():
    assert td1.strides[-1] == td1.dtype.itemsize
    assert td2.strides[-1] == td2.dtype.itemsize
    assert td2.strides[-2] == td2.dtype.itemsize * td2.shape[-1]


e0 = empty(td0)
e1 = empty(td1)
e2 = empty(td2)


def test_simple_tensors():
    assert e0.shape == td0.shape
    assert e1.shape == td1.shape
    assert e2.shape == td2.shape


td0_1 = td0.reaxe([ax.A])
e0_1 = tensorview(td0_1, e0)
td0_2 = td0.reaxe([ax.A, ax.B])
e0_2 = tensorview(td0_2, e0)

td1_2a = td1.reaxe([ax.A, ax.B])
td1_2b = td1.reaxe([ax.B, ax.A])
td1_2c = td1.reaxe([(ax.B, ax.C), ax.A])
e1_2a = tensorview(td1_2a, e1)
e1_2b = tensorview(td1_2b, e1)
e1_2c = tensorview(td1_2c, e1)

td2_2a = td2.reaxe([ax.B, ax.A])
td2_1a = td2.reaxe([[ax.A, ax.B]])
e2_2a = tensorview(td2_2a, e2)
e2_1a = tensorview(td2_1a, e2)


def test_views():
    val = 3

    assert e0_1.shape == td0_1.shape
    e0[()] = val
    for i in xrange(e1.shape[0]):
        assert e0_1[i] == val

    assert e0_2.shape == td0_2.shape
    for i in xrange(e2.shape[0]):
        for j in xrange(e2.shape[1]):
            assert e0_2[i, j] == val

    assert e1_2a.shape == (ax.A.length, ax.B.length)
    assert e1_2b.shape == (ax.B.length, ax.A.length)
    assert e1_2c.shape == (ax.B.length * ax.C.length, ax.A.length)

    for i in xrange(ax.A.length):
        e1[i] = val * i

    for i in xrange(ax.A.length):
        assert e1[i] == val * i
        for j in xrange(ax.B.length):
            assert e1_2a[i, j] == val * i
            assert e1_2b[j, i] == val * i
        for j in xrange(ax.B.length * ax.C.length):
            assert e1_2c[j, i] == val * i

    assert e2_2a.shape == (ax.B.length, ax.A.length)
    assert e2_1a.shape == (ax.A.length * ax.B.length,)

    def val3(i, j):
        return (i + 1) * (j + 2)

    for i in xrange(ax.A.length):
        for j in xrange(ax.B.length):
            e2[i, j] = val3(i, j)

    for i in xrange(ax.A.length):
        for j in xrange(ax.B.length):
            assert e2[i, j] == val3(i, j)
            assert e2_2a[j, i] == val3(i, j)
