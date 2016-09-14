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
from builtins import range

import ngraph.util.names as names
from ngraph.op_graph import arrayaxes

# Make some axes
ax = names.NameScope()

ax.A = arrayaxes.Axis(10)
ax.B = arrayaxes.Axis(15)
ax.C = arrayaxes.Axis(20)
ax.D = arrayaxes.Axis(25)


def test_axes_constructor_canonicalization():
    """ Test canonicalization of Axes constructor """
    a1 = arrayaxes.Axes(ax.A)
    a2 = arrayaxes.Axes([[ax.A]])
    assert a1 == a2


def test_axes_equal():
    """ Test axes == operator """
    a1 = arrayaxes.Axes([ax.A, ax.B, ax.C])
    a2 = arrayaxes.Axes([ax.A, ax.B, ax.C])
    assert a1 == a2


def to_nested_tuple(axes):
    """
    Recursively replace instances of FlattenedAxis with instances of type tuple.

    Arguments:
        axes: Axes object or iterable of Axis objects

    Returns:
        passes through axes objects with everything unchanged except
        FlattenedAxis are replaced with tuple
    """
    return tuple(
        to_nested_tuple(axis.axes) if isinstance(axis, arrayaxes.FlattenedAxis) else axis
        for axis in axes
    )


def test_canonicalize_axes():
    """ """
    def test(l, r):
        """
        TODO.

        Arguments:
          l: TODO
          r: TODO
        """
        a = arrayaxes.Axes(l)
        assert to_nested_tuple(a) == r, (
            'Failed. Original collection: {l}, axes: {a},'
            ' generated tuple: {t}, target: {r}'
        ).format(l=l, a=a, t=to_nested_tuple(a), r=r)

    test((), ())
    test([], ())
    test([ax.A], (ax.A,))
    test([ax.A, (ax.B,)], (ax.A, ax.B))
    test([ax.A, (ax.B, ax.C)], (ax.A, (ax.B, ax.C)))


def test_axes_ops():
    """TODO."""
    # Subtraction
    def test_sub(axes1, axes2, target):
        """
        TODO.

        Arguments:
          axes1: TODO
          axes2: TODO
          target: TODO

        Returns:

        """
        assert axes1 - axes2 == target

    test_sub(arrayaxes.Axes([ax.A, ax.B]), arrayaxes.Axes([ax.A]), arrayaxes.Axes([ax.B]))
    test_sub(arrayaxes.Axes([ax.A, ax.B]), arrayaxes.Axes([ax.B]), arrayaxes.Axes([ax.A]))

    # Combined axes length
    assert arrayaxes.FlattenedAxis(arrayaxes.Axes([ax.A, ax.B])).length \
        == ax.A.length * ax.B.length
    assert arrayaxes.Axes([ax.A, (ax.B, ax.C)]).lengths \
        == (ax.A.length, ax.B.length * ax.C.length)
    assert arrayaxes.FlattenedAxis(arrayaxes.Axes([ax.A, (ax.B, ax.C)])).length \
        == ax.A.length * ax.B.length * ax.C.length


def random(tensor_description):
    """
    return a ranom numpy array with dimension and dtype specified by
    tensor_description.

    Arguments:
        tensor_description: location of dimension and dtype specifications for
            returned array.
    """
    return np.random.random(
        tensor_description.shape
    ).astype(tensor_description.dtype)


def tensorview(td, nparr):
    """
    Returns a numpy array which whose buffer is nparr using the
    tensordescription in td

    Arguments:
        td TensorDescription: the description of the view of the nparr buffer
        nparr: the memory the np.array should use

    Returns:
      np.array view of nparr
    """
    return np.ndarray(
        shape=td.shape,
        dtype=td.dtype,
        buffer=nparr,
        offset=td.offset,
        strides=td.strides
    )


def test_reaxe_0d_to_1d():
    td = arrayaxes.TensorDescription(axes=())
    x = random(td)

    # create view of x
    x_view = tensorview(td.reaxe([ax.A]), x)

    # set x
    x[()] = 3

    # setting e also sets x_view
    assert x_view.shape == (ax.A.length,)
    assert np.all(x_view == 3)


def test_reaxe_0d_to_2d():
    td = arrayaxes.TensorDescription(axes=())
    x = random(td)

    x_view = tensorview(td.reaxe([ax.A, ax.B]), x)

    # set x
    x[()] = 3

    assert x_view.shape == (ax.A.length, ax.B.length)
    assert np.all(x_view == 3)


def test_simple_tensors():
    """
    tons of tests relating to reaxeing tensors.

    variables names have a postfix integer which represents the dimensionality
    of the value.  Views have x_y postfix which means they are y dimensional
    views of x dimensional buffers.

    I started refactoring into smaller pieces as seen in tests above, but
    stopped ...
    """
    # A simple vector
    td1 = arrayaxes.TensorDescription(axes=[ax.A])
    e1 = random(td1)

    td2 = arrayaxes.TensorDescription(axes=[ax.A, ax.B])
    e2 = random(td2)

    # Reaxes
    e1_1 = tensorview(td1.reaxe([ax.A, ax.B]), e1)
    e1_2 = tensorview(td1.reaxe([ax.B, ax.A]), e1)
    e1_3 = tensorview(td1.reaxe([(ax.B, ax.C), ax.A]), e1)

    e2_1 = tensorview(td2.reaxe(arrayaxes.Axes([ax.B, ax.A])), e2)
    e2_2 = tensorview(td2.reaxe(arrayaxes.Axes([ax.A, ax.B])), e2)
    e2_3 = tensorview(td2.reaxe(arrayaxes.Axes([arrayaxes.FlattenedAxis(td2.axes)])), e2)

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


def test_sliced_axis():
    a = arrayaxes.Axis(10)
    s = arrayaxes.SlicedAxis(a, slice(0, 5))
    assert s.length == 5


def test_sliced_axis_invalid():
    a = arrayaxes.Axis(10)
    s = arrayaxes.SlicedAxis(a, slice(5, 0))
    assert s.length == 0


def test_sliced_axis_none_end():
    a = arrayaxes.Axis(10)
    s = arrayaxes.SlicedAxis(a, slice(0, None))
    assert s.length == 10


def test_sliced_axis_negative():
    a = arrayaxes.Axis(10)
    s = arrayaxes.SlicedAxis(a, slice(5, 0, -1))
    assert s.length == 5


def test_sliced_axis_negative_invalid():
    a = arrayaxes.Axis(10)
    s = arrayaxes.SlicedAxis(a, slice(0, 5, -1))
    assert s.length == 0


def test_sliced_axis_flip():
    a = arrayaxes.Axis(10)
    s = arrayaxes.SlicedAxis(a, slice(None, None, -1))
    assert s.length == 10


def test_sliced_axis_invalid_step():
    a = arrayaxes.Axis(10)
    with pytest.raises(ValueError):
        arrayaxes.SlicedAxis(a, slice(0, 5, 2))


def test_sliced_batch_axis():
    """ slicing a batch axis should result in a batch axis """
    a = arrayaxes.Axis(10, batch=True)
    s = arrayaxes.SlicedAxis(a, slice(0, 5))
    assert s.batch is True
