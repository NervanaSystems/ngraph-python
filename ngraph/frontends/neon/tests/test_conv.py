import pytest
import numpy as np
import ngraph as ng
from ngraph.testing import executor
from ngraph.op_graph.axes import IncompatibleAxesError
from ngraph.frontends.neon import Convolution


# TODO: Remove these to conftest.py
@pytest.fixture(params=[1])
def input_size(request):
    return request.param


@pytest.fixture(params=[16])
def output_size(request):
    return request.param


@pytest.fixture(params=[4])
def batch_size(request):
    return request.param


@pytest.fixture
def width_axis(width):
    return ng.make_axis(length=width, name="W")


@pytest.fixture
def conv1d_placeholder(channel_axis, width_axis, batch_axis):
    return ng.placeholder((channel_axis, width_axis, batch_axis))


@pytest.fixture
def conv1d_no_channel_axis(width_axis, batch_axis):
    return ng.placeholder((width_axis, batch_axis))


@pytest.fixture
def spatial_onehot(input_size, width, batch_size):
    value = np.zeros((input_size, width, batch_size))
    value[:, width // 2, :] = 1
    return value


def test_causal_convolution(conv1d_placeholder, spatial_onehot, output_size, width):
    """ Test that causal convolutions only operate on leftward inputs"""
    conv_layer = Convolution((3, output_size), lambda x: 1, padding="causal")
    output = conv_layer(conv1d_placeholder)
    output_width = output.axes.find_by_name("W")[0].length
    assert  output_width == width, "Causal convolution output width != " \
                                   "input width: {} != {}".format(output_width, width)
    with executor(output, conv1d_placeholder) as comp:
        output_val = comp(spatial_onehot)
        # First 1 is at width // 2, so anything before that should be 0
        assert (output_val[:, :width // 2] == 0).all(), "Acausal outputs in causal convolution"


@pytest.mark.parametrize("stride", (1, 3))
def test_same_convolution(conv1d_placeholder, spatial_onehot, output_size, width, stride):
    """ Test that 'same' always results in out_size = np.ceil(in_size / stride) """
    conv_layer = Convolution((3, output_size), lambda x: 1, strides=stride, padding="same")
    output = conv_layer(conv1d_placeholder)
    output_width = output.axes.find_by_name("W")[0].length
    assert  output_width == np.ceil(width / float(stride)), ("Same convolution output width != " 
                                                             "ceil(input_width / stride): {} != "
                                                             "ceil({} / {})").format(output_width,
                                                                                     width,
                                                                                     stride)


def test_axis_preservation(conv1d_placeholder, output_size):
    """ Test that axes into a conv are the same as axes out"""
    conv_layer = Convolution((3, output_size), lambda x: 1)
    output = conv_layer(conv1d_placeholder)
    assert output.axes == conv1d_placeholder.axes, ("Output axes are not the same as input axes: "
                                                    "{} != {}").format(output.axes,
                                                                       conv1d_placeholder.axes)


def test_channel_axis_introduction(conv1d_no_channel_axis, output_size, channel_axis):
    """ Test that a channel axis is added when it doesn't exist in the input"""
    conv_layer = Convolution((3, output_size), lambda x: 1)
    output = conv_layer(conv1d_no_channel_axis)
    t_axes = conv1d_no_channel_axis.axes + channel_axis
    assert output.axes.is_equal_set(t_axes), ("Output axes are not input axes + channel axis:"
                                              "{} != {} + {}").format(output.axes,
                                                                      conv1d_no_channel_axis.axes,
                                                                      channel_axis)


def test_alternate_spatial_axes(conv1d_placeholder, output_size, width_axis):
    """ Test that spatial axis names are modifiable """
    width_axis.name = "time"
    assert len(conv1d_placeholder.axes.find_by_name("time")) == 1

    conv_layer = Convolution((3, output_size), lambda x: 1)
    with pytest.raises(IncompatibleAxesError):
        conv_layer(conv1d_placeholder)
    # As a dictionary
    output = conv_layer(conv1d_placeholder, spatial_axes={"W": "time"})
    assert output.axes == conv1d_placeholder.axes
    # As a tuple
    output = conv_layer(conv1d_placeholder, spatial_axes=("D", "H", "time"))
    assert output.axes == conv1d_placeholder.axes




def test_alternate_channel_axes(conv1d_placeholder ,output_size, channel_axis):
    """ Test that channel axis names are modifiable"""
    channel_axis.name = "channel"
    assert len(conv1d_placeholder.axes.find_by_name("channel")) == 1

    conv_layer = Convolution((3, output_size), lambda x: 1)
    with pytest.raises(IncompatibleAxesError):
        conv_layer(conv1d_placeholder)
    output = conv_layer(conv1d_placeholder, channel_axes="channel")
    assert output.axes == conv1d_placeholder.axes
