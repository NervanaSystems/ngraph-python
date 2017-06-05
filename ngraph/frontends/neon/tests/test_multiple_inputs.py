import pytest
import numpy as np
import ngraph as ng
from ngraph.frontends import neon


@pytest.fixture(params=[3])
def input_size(request):
    return request.param


def test_convolution(input_axis, spatial_axes):
    feature_axes = spatial_axes + input_axis

    batch_axis_1 = ng.make_axis(name='N', length=1)
    batch_axis_all = ng.make_axis(name='N', length=32)

    state_placeholder = ng.placeholder(feature_axes + [batch_axis_1])
    state_placeholder_all = ng.placeholder(feature_axes + [batch_axis_all])

    model = neon.Convolution((1, 1, 2), neon.XavierInit())

    def make_function(placeholder):
        computation = ng.computation(model(placeholder), placeholder)

        transformer = ng.transformers.make_transformer()
        return transformer.add_computation(computation)

    # make a computation using the batch size of 32, though it doesn't need to be used
    make_function(state_placeholder_all)
    # now make another computation with the batch size of 1.
    # the convolution fails because part of the graph is thinking that the batch size
    # is 32 and part of the graph is thinking it is a batch size of 1
    make_function(state_placeholder)(np.zeros(state_placeholder.axes.lengths))
