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
from builtins import range

import numpy as np
import pytest

import ngraph as ng
from ngraph.op_graph.axes import FlattenedAxis, TensorDescription, slice_axis
from ngraph.op_graph.axes import AxesMap, DuplicateAxisNames
from ngraph.testing import ExecutorFactory


def test_duplicate_axis():
    a = ng.make_axis(name='X')
    b = ng.make_axis(name='X')
    with pytest.raises(ValueError):
        ng.make_axes([a, b])


def test_duplicate_axis_different_length():
    a = ng.make_axis(1, name='N')
    b = ng.make_axis(2, name='N')
    with pytest.raises(ValueError) as e:
        ng.make_axes([a, b])

    # ensure the name of the axis appears in the exception
    assert 'N' in str(e)


# axes for testing
ax_A = ng.make_axis(2, name='A')
ax_B = ng.make_axis(3, name='B')
ax_C = ng.make_axis(4, name='C')

# axes for testing name matching behavior
ax_A_ = ng.make_axis(5, name='A')
ax_B_ = ng.make_axis(6, name='B')
ax_C_ = ng.make_axis(7, name='C')

# list of [axes_op_str, lhs_axes, rhs_axes, expected_res]
# currently Axis matches on name, while lengths is ignored
# so ax_A and ax_A_ is equivalent below
axes_ops_test_cases = [
    # add (list operation)
    ['__add__', [], [], []],
    ['__add__', [ax_A], [], [ax_A]],
    ['__add__', [ax_A_], [], [ax_A]],
    ['__add__', [ax_A], [ax_B], [ax_A, ax_B]],
    ['__add__', [ax_A_], [ax_B_], [ax_A, ax_B]],
    # add (list operation, test exception)
    ['__add__', [ax_A], [ax_A], ValueError],
    ['__add__', [ax_A], [ax_A_], ValueError],
    ['__add__', [ax_A], [ax_A_, ax_B], ValueError],
    # difference (set operation, ordered)
    ['__sub__', [], [], []],
    ['__sub__', [], [ax_A], []],
    ['__sub__', [ax_A], [], [ax_A]],
    ['__sub__', [ax_A, ax_B], [ax_B], [ax_A]],
    ['__sub__', [ax_A, ax_B], [ax_B_], [ax_A]],
    ['__sub__', [ax_A, ax_B], [ax_A], [ax_B]],
    ['__sub__', [ax_A, ax_B], [ax_A_], [ax_B]],
    ['__sub__', [ax_A, ax_B], [ax_B, ax_A], []],
    ['__sub__', [ax_A, ax_B], [ax_B_, ax_A_], []],
    # union (set operation, ordered)
    ['__or__', [], [], []],
    ['__or__', [], [ax_A], [ax_A]],
    ['__or__', [ax_A], [], [ax_A]],
    ['__or__', [ax_A], [ax_B], [ax_A, ax_B]],
    ['__or__', [ax_A], [ax_A_], [ax_A]],
    ['__or__', [ax_A], [ax_A_], [ax_A_]],
    # intersection (set operation, ordered)
    ['__and__', [], [], []],
    ['__and__', [], [ax_A], []],
    ['__and__', [ax_A], [], []],
    ['__and__', [ax_A], [ax_B], []],
    ['__and__', [ax_A, ax_B], [ax_B, ax_C], [ax_B]],
    ['__and__', [ax_A, ax_B_], [ax_B, ax_C], [ax_B]],
    # equal (list operation, ordered)
    ['__eq__', [], [], True],
    ['__eq__', [ax_A], [], False],
    ['__eq__', [ax_A, ax_B], [ax_B, ax_A], False],
    ['__eq__', [ax_A, ax_B], [ax_B_, ax_A_], False],
    ['__eq__', [ax_A, ax_B], [ax_A_, ax_B], True],
    ['__eq__', [ax_A, ax_B], [ax_A_, ax_B_], True],
    # not equal (list operation, ordered)
    ['__ne__', [], [], False],
    ['__ne__', [ax_A], [], True],
    ['__ne__', [ax_A, ax_B], [ax_B, ax_A], True],
    ['__ne__', [ax_A, ax_B], [ax_B_, ax_A_], True],
    ['__ne__', [ax_A, ax_B], [ax_A_, ax_B], False],
    ['__ne__', [ax_A, ax_B], [ax_A_, ax_B_], False],
    # subset
    ['is_sub_set', [], [], True],
    ['is_sub_set', [ax_A], [], False],
    ['is_sub_set', [], [ax_A], True],
    ['is_sub_set', [ax_A_], [ax_A], True],
    ['is_sub_set', [ax_A, ax_B], [ax_B, ax_A], True],
    ['is_sub_set', [ax_A, ax_B], [ax_B_, ax_A_], True],
    # superset
    ['is_super_set', [], [], False],
    ['is_super_set', [ax_A], [], True],
    ['is_super_set', [], [ax_A], False],
    ['is_super_set', [ax_A_], [ax_A], False],
    ['is_super_set', [ax_A, ax_B], [ax_B, ax_A], False],
    ['is_super_set', [ax_A, ax_B], [ax_B_, ax_A_], False],
    # set equal
    ['is_equal_set', [], [], True],
    ['is_equal_set', [ax_A], [], False],
    ['is_equal_set', [ax_A], [ax_A], True],
    ['is_equal_set', [ax_A], [ax_A_], True],
    ['is_equal_set', [ax_A, ax_B], [ax_B_, ax_A_], True],
    # set not equal
    ['is_not_equal_set', [], [], False],
    ['is_not_equal_set', [ax_A], [], True],
    ['is_not_equal_set', [ax_A], [ax_A], False],
    ['is_not_equal_set', [ax_A], [ax_A_], False],
    ['is_not_equal_set', [ax_A, ax_B], [ax_B_, ax_A_], False],
]


@pytest.mark.parametrize("test_cases", axes_ops_test_cases)
def test_axes_ops(test_cases):
    # unpack test case
    axes_op_str, lhs_axes, rhs_axes, expected_res = test_cases
    lhs_axes = ng.make_axes(lhs_axes)
    rhs_axes = ng.make_axes(rhs_axes)
    if isinstance(expected_res, list):
        expected_res = ng.make_axes(expected_res)

    # check results against expected_res
    if expected_res is ValueError:
        with pytest.raises(ValueError):
            getattr(lhs_axes, axes_op_str)(rhs_axes)
    else:
        res = getattr(lhs_axes, axes_op_str)(rhs_axes)
        if res != expected_res:
            raise ValueError("%s operation with %s and %s, "
                             "expected result %s, but actually get %s"
                             % (axes_op_str, lhs_axes, rhs_axes, expected_res, res))


def random(tensor_description):
    """
    return a random numpy array with dimension and dtype specified by
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
    td = TensorDescription(())
    x = random(td)

    # create view of x
    x_view = tensorview(td.broadcast([ax_A]), x)

    # set x
    x[()] = 3

    # setting e also sets x_view
    assert x_view.shape == (ax_A.length,)
    assert np.all(x_view == 3)


def test_reaxe_0d_to_2d():
    td = TensorDescription(axes=())
    x = random(td)

    x_view = tensorview(td.broadcast([ax_A, ax_B]), x)

    # set x
    x[()] = 3

    assert x_view.shape == (ax_A.length, ax_B.length)
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
    td1 = TensorDescription(axes=[ax_A])
    e1 = random(td1)

    td2 = TensorDescription(axes=[ax_A, ax_B])
    e2 = random(td2)

    # Reaxes
    e1_1 = tensorview(td1.broadcast([ax_A, ax_B]), e1)
    e1_2 = tensorview(td1.broadcast([ax_B, ax_A]), e1)
    e1_3 = tensorview(td1.broadcast([(ax_B, ax_C), ax_A]), e1)

    e2_1 = tensorview(td2.broadcast([ax_B, ax_A]), e2)
    e2_2 = tensorview(td2.broadcast([ax_A, ax_B]), e2)
    e2_3 = tensorview(td2.flatten((
        FlattenedAxis((ax_A, ax_B)),
    )), e2_2)

    assert e1_1.shape == (ax_A.length, ax_B.length)
    assert e1_2.shape == (ax_B.length, ax_A.length)

    for i in range(ax_A.length):
        e1_1[i] = i

    for i in range(ax_A.length):
        assert e1[i] == i
        for j in range(ax_B.length):
            assert e1_1[i, j] == i
            assert e1_2[j, i] == i
        for j in range(ax_B.length * ax_C.length):
            assert e1_3[j, i] == i

    def val2(i, j):
        return (i + 1) * (j + 2)

    for i in range(ax_A.length):
        for j in range(ax_B.length):
            e2[i, j] = val2(i, j)

    for i in range(ax_A.length):
        for j in range(ax_B.length):
            assert e2_1[j, i] == val2(i, j)
            assert e2_2[i, j] == val2(i, j)
            assert e2_3[i * ax_B.length + j] == val2(i, j)


def test_sliced_axis():
    a = ng.make_axis(10)
    s = slice_axis(a, slice(0, 5))
    assert s.length == 5


def test_sliced_axis_invalid():
    a = ng.make_axis(10)
    s = slice_axis(a, slice(5, 0))
    assert s.length == 0


def test_sliced_axis_none_end():
    a = ng.make_axis(10)
    s = slice_axis(a, slice(0, None))
    assert s.length == 10


def test_sliced_axis_negative():
    a = ng.make_axis(10)
    s = slice_axis(a, slice(5, 0, -1))
    assert s.length == 5


def test_sliced_axis_negative_invalid():
    a = ng.make_axis(10)
    s = slice_axis(a, slice(0, 5, -1))
    assert s.length == 0


def test_sliced_axis_flip():
    a = ng.make_axis(10)
    s = slice_axis(a, slice(None, None, -1))
    assert s.length == 10


def test_sliced_axis_invalid_step():
    a = ng.make_axis(10)
    with pytest.raises(ValueError):
        slice_axis(a, slice(0, 5, 2))


def test_sliced_batch_axis():
    """ slicing a batch axis should result in a batch axis """
    a = ng.make_axis(10, name='N')
    s = slice_axis(a, slice(0, 5))
    assert s.is_batch is True


def test_sliced_recurrent_axis():
    """ slicing a recurrent axis should result in a recurrent axis """
    a = ng.make_axis(10, name='R')
    s = slice_axis(a, slice(0, 5))
    assert s.is_recurrent is True


def test_sliced_axis_roles():
    """ slicing an axis should result in the same roles as the parent axis """
    role1 = ng.make_axis_role()
    role2 = ng.make_axis_role()
    a = ng.make_axis(10, roles=[role1, role2])
    s = slice_axis(a, slice(0, 5))
    assert all(r in s.roles for r in a.roles)


def test_idempotent_axes_a():
    """
    Test test axes transformations with autodiff, case a, reference test
    """
    with ExecutorFactory() as ex:
        axes = ng.make_axes([ng.make_axis(3), ng.make_axis(1)])

        w = ng.variable(axes, initial_value=np.ones((3, 1)))
        result = w + w

        result = ng.cast_axes(result, axes)
        cost = ng.sum(result, reduction_axes=axes)
        grad = ng.deriv(cost, w)

        grad_comp = ex.executor(grad)
        cost_comp = ex.executor(cost)

        assert cost_comp() == 6.0
        assert np.array_equal(grad_comp(), np.ones((3, 1)) * 2.)


def test_idempotent_axes_b():
    """
    Test test axes transformations with autodiff, case b, with broadcast applied
    to the same tensor
    """
    with ExecutorFactory() as ex:
        axes = ng.make_axes([ng.make_axis(3), ng.make_axis(1)])

        w = ng.variable(axes, initial_value=np.ones((3, 1)))
        l = ng.broadcast(w, axes)
        r = ng.broadcast(w, axes)
        result = ng.add(l, r)

        result = ng.cast_axes(result, axes)
        cost = ng.sum(result, reduction_axes=axes)
        grad = ng.deriv(cost, w)

        grad_comp = ex.executor(grad)
        cost_comp = ex.executor(cost)

        assert cost_comp() == 6.0
        assert np.array_equal(grad_comp(), np.ones((3, 1)) * 2.)


def test_idempotent_axes_c():
    """
    Test test axes transformations with autodiff, case c, with broadcast,
    slice, cast and dim-shuffle
    """
    with ExecutorFactory() as ex:
        axes = ng.make_axes([ng.make_axis(3), ng.make_axis(1)])
        result_axes = [ng.make_axis(length=axis.length) for axis in axes]

        # variable
        w = ng.variable(axes, initial_value=np.ones((3, 1)))

        # broadcast l / r, introducing dummy length 1 axes
        l = ng.broadcast(w, axes)
        r = ng.broadcast(w, axes)

        # slice
        axes_slice = [slice(None, None, None), slice(None, None, None)]
        l_sliced = ng.tensor_slice(l, axes_slice)
        r_sliced = ng.tensor_slice(r, axes_slice)

        # cast r
        r_sliced_casted = ng.cast_axes(r_sliced, axes)

        # perform add
        result = ng.add(l_sliced, r_sliced_casted)

        # cast / dimshuffle
        result = ng.cast_axes(result, result_axes)
        result = ng.axes_with_order(result, result_axes)

        # cost and grad
        cost = ng.sum(result, reduction_axes=result.axes)
        grad = ng.deriv(cost, w)

        grad_comp = ex.executor(grad)
        cost_comp = ex.executor(cost)

        cost_comp_ng = cost_comp()
        grad_comp_ng = grad_comp()
        grad_comp_np = np.ones((3, 1)) * 2.
        assert cost_comp_ng == 6.0
        assert np.array_equal(grad_comp_ng, grad_comp_np)


def test_scalar_broadcast():
    """
    Test broadcasting a scalar into a tensor
    """
    with ExecutorFactory() as ex:
        x_axes = ng.make_axes()
        broadcast_axes = ng.make_axes([ng.make_axis(2), ng.make_axis(3)])
        x = ng.constant(1., axes=x_axes)
        z = ng.broadcast(x, axes=broadcast_axes)
        z_comp = ex.executor(z)
        assert np.array_equal(z_comp(), np.ones(broadcast_axes.lengths))


def test_duplicate_axis_names():
    with pytest.raises(DuplicateAxisNames) as e:
        AxesMap({'aaa': 'zzz', 'bbb': 'zzz', 'ccc': 'yyy'})

    assert e.value.duplicate_axis_names == {
        'zzz': set(['aaa', 'bbb']),
    }


def test_invalid_axes_map_message():
    with pytest.raises(ValueError) as exc_info:
        AxesMap({'aaa': 'zzz', 'bbb': 'zzz', 'ccc': 'yyy'})

    e = exc_info.value

    # check that offending names are listed in the error message
    assert 'aaa' in str(e)
    assert 'bbb' in str(e)
    assert 'zzz' in str(e)

    # check that non-offending names are not listed in the error message
    assert 'ccc' not in str(e)
    assert 'yyy' not in str(e)


def test_axes_map():
    """
    map from Axes([aaa, bbb]) to Axes([zzz, bbb]) via AxesMap {aaa: zzz}
    """
    a = ng.make_axis(1, name='aaa')
    b = ng.make_axis(2, name='bbb')
    z = ng.make_axis(1, name='zzz')

    axes_before = ng.make_axes([a, b])
    axes_after = ng.make_axes([z, b])

    axes_map = AxesMap({a.name: z.name})

    assert axes_after == axes_map.map_axes(axes_before)


def test_axes_map_immutable():
    axes_map = AxesMap({})

    with pytest.raises(TypeError):
        axes_map['x'] = 'y'


def test_axes_map_init_from_axes():
    axes_map = AxesMap({ng.make_axis(1, name='aaa'): ng.make_axis(1, name='zzz')})

    assert axes_map['aaa'] == 'zzz'
