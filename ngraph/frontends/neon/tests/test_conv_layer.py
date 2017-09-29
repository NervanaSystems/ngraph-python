import pytest
import numpy as np
from contextlib import closing
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.testing import executor
from ngraph.frontends.common import utils
from ngraph.op_graph.axes import IncompatibleAxesError
from ngraph.frontends.neon import Convolution, Deconvolution
from ngraph.frontends.neon import ConstantInit, Rectlin, GaussianInit


def reference_conv1d(inputs, filters, activation, strides=1, padding=0):
    # for now:
    assert strides == 1
    assert padding == 0
    # inputs: features, time steps (conv axis), batch size
    # filters: input feature dimension/channels, T=1, R=filter_width, S=1, K=num_filters
    # result: K, 1, time_steps - S + 1, 1, batch size
    filters = np.squeeze(filters)  # input channels, filter_width, num_filters
    feature_dimension, time_steps_in, batch_size = inputs.shape
    filter_width = filters.shape[1]
    K = filters.shape[-1]
    time_steps_out = time_steps_in - filter_width + 1
    result = np.zeros((K, time_steps_out, batch_size))
    # TODO: refactor to make this more efficient
    for t in range(time_steps_out):
        for k in range(K):
            for n in range(batch_size):
                result[k, t, n] = np.sum(inputs[:, t:t+filter_width, n] * filters[:, :, k])
    result = activation(result)
    # expand dimensions from K, time_steps, batch_size to (K, 1, time_steps, 1, batch_size)
    result = np.expand_dims(np.expand_dims(result, axis=1), axis=3)
    return result


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
    assert output_width == width, "Causal convolution output width != " \
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
    assert output_width == np.ceil(width / float(stride)), ("Same convolution output width != "
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


def test_alternate_channel_axes(conv1d_placeholder, output_size, channel_axis):
    """ Test that channel axis names are modifiable"""
    channel_axis.name = "channel"
    assert len(conv1d_placeholder.axes.find_by_name("channel")) == 1

    conv_layer = Convolution((3, output_size), lambda x: 1)
    with pytest.raises(IncompatibleAxesError):
        conv_layer(conv1d_placeholder)
    output = conv_layer(conv1d_placeholder, channel_axes="channel")
    assert output.axes == conv1d_placeholder.axes


@pytest.mark.parametrize('filter_width', [3])
@pytest.mark.parametrize('num_filters', [2])
@pytest.mark.parametrize('strides', [1])
@pytest.mark.parametrize('padding', [0])
@pytest.mark.parametrize('time_steps', [5])
@pytest.mark.parametrize('feature_dimension', [4])
@pytest.mark.parametrize('batch_size', [2])
def test_conv1d(transformer_factory, filter_width, num_filters, strides, padding,
                time_steps, feature_dimension, batch_size):

    dilation = 1  # reference conv does not support dilation

    F = ng.make_axis(name='F', length=feature_dimension)
    REC = ng.make_axis(name='REC', length=time_steps)
    N = ng.make_axis(name='N', length=batch_size)
    in_axes = ng.make_axes([F, REC, N])

    inputs = ng.placeholder(axes=in_axes)
    input_vals = np.random.randn(*in_axes.lengths)

    filter_init = GaussianInit()

    conv1d = Convolution((filter_width, num_filters), filter_init,
                           strides=strides, padding=padding, dilation=dilation,
                           bias_init=None, activation=Rectlin(), batch_norm=None)

    result_op = conv1d(inputs, channel_axes='F', spatial_axes={'W': 'REC'})

    with closing(ngt.make_transformer()) as transformer:
        result_comp = transformer.add_computation(ng.computation(result_op, inputs))
        filter_vals = transformer.add_computation(ng.computation(conv1d.conv.W))()

        result_ng = result_comp(input_vals)
        result_np = np.squeeze(reference_conv1d(input_vals, filter_vals, lambda x: np.maximum(0, x)))
        ng.testing.assert_allclose(result_ng, result_np)


# TODO: add other configurations?
@pytest.mark.transformer_dependent
@pytest.config.flex_disabled(reason="#1841, deconv is not yet supported by flex")
def test_deconv():
    """
    basic test of deconv fprop.
    ngraph/tests/test_conv.py tests ng.deconvolution bprop
    """

    # filter params
    R, S = 5, 5
    fshape = (R, S, 1)
    strides = 2
    filter_val_nz = np.arange(1, R * S + 1).reshape(R, S)
    filter_val = np.zeros(fshape)
    filter_val[:, :, 0] = filter_val_nz

    deconv = Deconvolution(fshape,
                           filter_init=ConstantInit(filter_val),
                           strides=strides,
                           padding=0,
                           dilation=1)

    N = ng.make_axis(name='N', length=1)  # batch
    image_shape = (1, 8, 8)  # CHW
    image_axes = ng.make_axes([ng.make_axis(name=nm, length=l)
                               for nm, l in zip('CHW', image_shape)])
    image_axes |= N
    image = ng.placeholder(axes=image_axes)

    output = deconv(image)

    with closing(ngt.make_transformer()) as transformer:
        comp = transformer.add_computation(ng.computation(output, image))
        input_val = np.zeros(image_shape + (N.length, ), dtype=float)
        input_val[0, 0, 0] = 1
        input_val[0, 5, 5] = 1
        input_val[0, 7, 7] = 1
        result = comp(input_val)
        feature_map = np.squeeze(result)

        assert (feature_map[:5, :5] == filter_val_nz).all()

        result2 = filter_val_nz.copy()
        result2[-1, -1] = 26
        assert (feature_map[10:15, 10:15] == result2).all()

        result3 = filter_val_nz.copy()
        result3[0, 0] = 26
        assert (feature_map[-5:, -5:] == result3).all()


@pytest.mark.parametrize("input_size", (10, 25))
@pytest.mark.parametrize("filter_size", (3, 4))
@pytest.mark.parametrize("padding", ((0, 0), (3, 4)))
@pytest.mark.parametrize("stride", (1, 3))
def test_conv_inverts_deconv(transformer_factory, input_size, filter_size, padding, stride):
    """ Test that conv and deconv are inverse operations given the same parameters"""

    # convolutions whose output size are not an even multiple of stride cannot be exactly inverted
    a = (input_size + sum(padding) - filter_size) % stride
    conv_output = utils.conv_output_dim(input_size, filter_size, padding, stride)
    deconv_output = utils.deconv_output_dim(conv_output, filter_size, padding, stride)

    assert deconv_output == (input_size - a), ("Convolution and Deconvolution do not invert:\n"
                                               "output ({}) != input ({}) - a ({})\n"
                                               "filter: {}, padding: {}, stride: {}"
                                               ).format(deconv_output, input_size, a,
                                                        filter_size, padding, stride)
