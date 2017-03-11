import pytest

import ngraph as ng
from ngraph.frontends.neon import axis


@pytest.fixture()
def axis_a():
    return ng.make_axis(1, name='a')


def test_axis_is_not_shadow(axis_a):
    assert not axis.is_shadow_axis(axis_a)


def test_shadow_axis_is_shadow_axis(axis_a):
    assert axis.is_shadow_axis(axis.make_shadow_axis(axis_a))
