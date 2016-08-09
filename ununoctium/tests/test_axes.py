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
from builtins import range

import geon.op_graph.names as names
from geon.op_graph import arrayaxes
from geon.transformers.nptransform import NumPyTransformer

# Make some axes
ax = names.NameScope()

ax.A = arrayaxes.Axis(10)
ax.B = arrayaxes.Axis(15)
ax.C = arrayaxes.Axis(20)
ax.D = arrayaxes.Axis(25)


def test_axes_equal():
    a1 = arrayaxes.Axes(ax.A, ax.B, ax.C)
    a2 = arrayaxes.Axes(ax.A, ax.B, ax.C)
    assert a1 == a2


def to_nested_tuple(axes):
    """ recursively replace instances of AxesAxis with instances of type tuple. """
    return tuple(
        to_nested_tuple(axis.axes) if isinstance(axis, arrayaxes.AxesAxis) else axis
        for axis in axes
    )


def test_canonicalize_axes():
    """

    """
    def test(l, r):
        a = arrayaxes.Axes(*l)
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
    # Subtraction
    def test_sub(axes1, axes2, target):
        assert arrayaxes.AxisIDTuple.sub(
            axes1.as_axis_ids(),
            axes2.as_axis_ids()).as_axes() == target
    test_sub(arrayaxes.Axes(ax.A, ax.B), arrayaxes.Axes(ax.A,), arrayaxes.Axes(ax.B,))
    test_sub(arrayaxes.Axes(ax.A, ax.B), arrayaxes.Axes(ax.B,), arrayaxes.Axes(ax.A,))

    # Combined axes length
    assert arrayaxes.AxesAxis(arrayaxes.Axes(ax.A, ax.B,)).length \
        == ax.A.length * ax.B.length
    assert arrayaxes.Axes(ax.A, (ax.B, ax.C)).lengths \
        == (ax.A.length, ax.B.length * ax.C.length)
    assert arrayaxes.AxesAxis(arrayaxes.Axes(ax.A, (ax.B, ax.C))).length \
        == ax.A.length * ax.B.length * ax.C.length


def empty(td):
    return np.empty(td.shape, td.dtype)


def tensorview(td, nparr):
    return np.ndarray(shape=td.shape, dtype=td.dtype,
                      buffer=nparr, offset=td.offset,
                      strides=td.strides)


def test_simple_tensors():
    transformer = NumPyTransformer()
    # A scalar
    td0 = arrayaxes.TensorDescription(
        axes=(), transformer=transformer
    )
    e0 = empty(td0)

    # A simple vector
    td1 = arrayaxes.TensorDescription(
        axes=[ax.A], transformer=transformer
    )
    e1 = empty(td1)

    td2 = arrayaxes.TensorDescription(
        axes=[ax.A, ax.B], transformer=transformer
    )
    e2 = empty(td2)

    td3 = arrayaxes.TensorDescription(
        axes=(ax.D, ax.D), transformer=transformer
    )
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

    td2_1 = td2.reaxe(arrayaxes.Axes(ax.B, ax.A))
    e2_1 = tensorview(td2_1, e2)
    td2_2 = td2.reaxe(arrayaxes.Axes(ax.A, ax.B))
    e2_2 = tensorview(td2_2, e2)
    td2_3 = td2.reaxe(arrayaxes.Axes(arrayaxes.AxesAxis(td2.axes)))
    e2_3 = tensorview(td2_3, e2)

    td3_1 = td3.reaxe_with_axis_ids(arrayaxes.AxisIDTuple(
        arrayaxes.AxisID(ax.D, 1),
        arrayaxes.AxisID(ax.D, 0)
    ))
    e3_1 = tensorview(td3_1, e3)

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
