import pytest
import ngraph as ng


@pytest.fixture(params=[10])
def height(request):
    return request.param


@pytest.fixture(params=[10])
def width(request):
    return request.param


@pytest.fixture(params=[0])
def extra_feature_axes(request):
    return [ng.make_axis(length=1, name="feature_{}".format(ii)) for ii in range(request.param)]


@pytest.fixture
def batch_axis(batch_size):
    return ng.make_axis(length=batch_size, name='N')


@pytest.fixture
def recurrent_axis(sequence_length):
    return ng.make_axis(length=sequence_length, name='REC')


@pytest.fixture
def input_axis(input_size):
    return ng.make_axis(length=input_size, name='input')


@pytest.fixture
def output_axis(output_size):
    return ng.make_axis(length=output_size, name="hidden")


@pytest.fixture
def spatial_axes(height, width):
    H = ng.make_axis(length=height, name="height")
    W = ng.make_axis(length=width, name="width")
    return ng.make_axes([H, W])


@pytest.fixture
def channel_axis(input_size):
    return ng.make_axis(length=input_size, name="C")


@pytest.fixture
def input_placeholder(input_axis, batch_axis, extra_feature_axes):
    return ng.placeholder(extra_feature_axes + [input_axis, batch_axis])


@pytest.fixture
def recurrent_input(input_axis, recurrent_axis, batch_axis, extra_feature_axes):
    return ng.placeholder(extra_feature_axes + [input_axis, recurrent_axis, batch_axis])


@pytest.fixture
def conv_input_placeholder(channel_axis, spatial_axes, batch_axis):
    axes = ng.make_axes(channel_axis) | spatial_axes | batch_axis
    return ng.placeholder(axes)
