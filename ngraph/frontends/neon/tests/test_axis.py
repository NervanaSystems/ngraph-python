import pytest

import ngraph as ng
from ngraph.op_graph.axes import IncompatibleAxesError
from ngraph.frontends.neon.axis import is_shadow_axis, make_shadow_axis, reorder_spatial_axes


@pytest.fixture()
def axis_a():
    return ng.make_axis(1, name='a')


@pytest.fixture()
def CDHWN():
    return ng.make_axes([
        ng.make_axis(3, name='C'),
        ng.make_axis(4, name='D'),
        ng.make_axis(5, name='H'),
        ng.make_axis(6, name='W'),
        ng.make_axis(8, name='N')
    ])


def test_axis_is_not_shadow(axis_a):
    assert not is_shadow_axis(axis_a)


def test_shadow_axis_is_shadow_axis(axis_a):
    assert is_shadow_axis(make_shadow_axis(axis_a))


def test_reorder_spatial_no_batch(CDHWN):
    tensor = ng.placeholder(CDHWN[0:2])
    with pytest.raises(ValueError):
        reorder_spatial_axes(tensor, "C", ("D", "H", "W"))


def test_reorder_spatial_no_channel(CDHWN):
    tensor = ng.placeholder(CDHWN[-2:])
    new_axes = reorder_spatial_axes(tensor, "C", ("D", "H", "W")).axes
    assert len(new_axes) == 5
    assert new_axes[0].name == 'C'
    assert new_axes[0].length == 1


def test_reorder_spatial_no_spatial(CDHWN):
    tensor = ng.placeholder([CDHWN[0], CDHWN[-1]])
    with pytest.raises(IncompatibleAxesError):
        reorder_spatial_axes(tensor, "C", ("D", "H", "W"))


def test_reorder_spatial_single_spatial(CDHWN):
    # Reorder to NCH
    tensor = ng.placeholder([CDHWN[-1], CDHWN[0], CDHWN[2]])
    new_axes = reorder_spatial_axes(tensor, "C", ("D", "H", "W")).axes
    assert new_axes == CDHWN
    assert new_axes[1].length == 1  # D has been added with length 1
    assert new_axes[3].length == 1  # W has been added with length 1


def test_reorder_spatial_double_spatial(CDHWN):
    # Reorder to NCWH
    tensor = ng.placeholder([CDHWN[-1], CDHWN[0], CDHWN[3], CDHWN[2]])
    new_axes = reorder_spatial_axes(tensor, "C", ("D", "H", "W")).axes
    assert new_axes == CDHWN
    assert new_axes[1].length == 1  # D has been added with length 1


def test_reorder_spatial_triple_spatial(CDHWN):
    # Reorder to NCWHD
    tensor = ng.placeholder([CDHWN[-1], CDHWN[0], CDHWN[3], CDHWN[2], CDHWN[1]])
    new_axes = reorder_spatial_axes(tensor, "C", ("D", "H", "W")).axes
    assert new_axes == CDHWN


def test_reorder_spatial_toomany_spatial(CDHWN, axis_a):
    tensor = ng.placeholder(CDHWN + axis_a)
    with pytest.raises(IncompatibleAxesError):
        reorder_spatial_axes(tensor, "C", ("D", "H", "W"))
