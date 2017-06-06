import pytest
import numpy as np
import ngraph as ng
from ngraph.frontends import neon
from ngraph.testing import ExecutorFactory


@pytest.fixture(params=[3])
def input_size(request):
    return request.param


def test_convolution(input_axis, spatial_axes):
    """
    this test ensures that we can share filter waits for convolution ops with
    different batch sizes.
    """
    feature_axes = spatial_axes + input_axis

    batch_axis_1 = ng.make_axis(name='N', length=1)
    batch_axis_all = ng.make_axis(name='N', length=32)

    state_placeholder_1 = ng.placeholder(feature_axes + [batch_axis_1])
    state_placeholder_all = ng.placeholder(feature_axes + [batch_axis_all])

    model = neon.Convolution((1, 1, 2), neon.XavierInit())

    with ExecutorFactory() as ex:
        ex.executor(model(state_placeholder_1), state_placeholder_1)
        computation = ex.executor(
            model(state_placeholder_all), state_placeholder_all
        )

        computation(np.zeros(state_placeholder_all.axes.lengths))
