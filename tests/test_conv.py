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

import ngraph as ng
from ngraph.util.utils import executor
from ngraph.util.utils import RandomTensorGenerator
from ngraph.op_graph.axes import spatial_axis
from ngraph.frontends.neon import ax, ar
from neon import NervanaObject
from neon.backends import gen_backend
from neon.layers.layer import Convolution
import pytest

rng = RandomTensorGenerator(0, np.float32)


NervanaObject.be = gen_backend()


class DummyDeltaBuffers(object):
    """
    Dummy class for delta buffers needed by neon
    """
    def __init__(self):
        self.buffers = [None]


def test_wrong_filters_shape_length():
    """
    test wrong filters shape length
    """
    padding = dict(pad_d=0, pad_h=0, pad_w=0)
    strides = dict(str_d=1, str_h=1, str_w=1)
    conv_params = padding.copy()
    conv_params.update(strides)

    ax_i = ng.make_axes([ax.C, ax.D, ax.H, ax.W, ax.N])
    ax_f = ng.make_axes([ax.C, ax.T, ax.R, ax.S])

    inputs = ng.placeholder(ax_i)
    filters = ng.placeholder(ax_f)

    with pytest.raises(ValueError) as exinfo:
        ng.convolution(conv_params, inputs, filters, {})
    assert str(exinfo.value) == 'convolution filter shape must be length 5, found {}'\
        .format(len(ax_f))


def test_wrong_input_shape_length():
    """
    test wrong input shape length
    """
    padding = dict(pad_d=0, pad_h=0, pad_w=0)
    strides = dict(str_d=1, str_h=1, str_w=1)
    conv_params = padding.copy()
    conv_params.update(strides)

    ax_i = ng.make_axes([ax.C, ax.D, ax.H, ax.W])
    ax_f = ng.make_axes([ax.C, ax.T, ax.R, ax.S, ax.K])

    inputs = ng.placeholder(ax_i)
    filters = ng.placeholder(ax_f)

    with pytest.raises(ValueError) as exinfo:
        ng.convolution(conv_params, inputs, filters, {})
    assert str(exinfo.value) == 'convolution input shape must be length 5, found {}'\
        .format(len(ax_i))


def test_first_axes_not_same():
    """
    test first axes are not the same
    """
    padding = dict(pad_d=0, pad_h=0, pad_w=0)
    strides = dict(str_d=1, str_h=1, str_w=1)
    conv_params = padding.copy()
    conv_params.update(strides)

    ax_i = ng.make_axes([ax.D, ax.C, ax.H, ax.W, ax.N])
    ax_f = ng.make_axes([ax.C, ax.T, ax.R, ax.S, ax.K])

    inputs = ng.placeholder(ax_i)
    filters = ng.placeholder(ax_f)

    with pytest.raises(ValueError) as exinfo:
        ng.convolution(conv_params, inputs, filters, {})
    assert str(exinfo.value) == 'the first axis in input {inputs} and filter {filters} ' \
        'are not the same.'.format(
            inputs=inputs.axes[0],
            filters=filters.axes[0])


def test_wrong_number_of_batch_axes_at_input():
    """
    test wrong number of batch axes at input
    """
    padding = dict(pad_d=0, pad_h=0, pad_w=0)
    strides = dict(str_d=1, str_h=1, str_w=1)
    conv_params = padding.copy()
    conv_params.update(strides)

    C = 3
    D = 1
    ax_C = ng.make_axis(length=C, name='C', batch=True)
    ax_D = ng.make_axis(length=D, name='D', batch=True)

    ax_i = ng.make_axes([ax_C, ax_D, ax.H, ax.W, ax.N])
    ax_f = ng.make_axes([ax_C, ax.T, ax.R, ax.S, ax.K])

    inputs = ng.placeholder(axes=ax_i)
    filters = ng.placeholder(ax_f)

    with pytest.raises(ValueError) as exinfo:
        ng.convolution(conv_params, inputs, filters, {})

    assert str(exinfo.value) == "Input must have one batch axis.  Found {n_batch_axes} " \
        "batch axes: {batch_axes} Found {n_sample_axes} sample axes: {sample_axes}.".format(
            n_batch_axes=len(inputs.axes.batch_axes()),
            batch_axes=inputs.axes.batch_axes(),
            n_sample_axes=len(inputs.axes.sample_axes()),
            sample_axes=inputs.axes.sample_axes())


def test_convolution(transformer_factory):
    """
    test convolution forward path
    """
    N = 128
    C, K = 3, 8
    D, T = 1, 1
    H = W = 32
    R = S = 2

    padding = dict(pad_d=0, pad_h=0, pad_w=0)
    strides = dict(str_d=1, str_h=1, str_w=1)
    conv_params = padding.copy()
    conv_params.update(strides)

    ax_i = ng.make_axes([ax.C, ax.D, ax.H, ax.W, ax.N])
    ax_f = ng.make_axes([ax.C, ax.T, ax.R, ax.S, ax.K])
    ax_i.set_shape((C, D, H, W, N))
    ax_f.set_shape((C, T, R, S, K))
    ax_o = ng.make_axes([
        ng.make_axis(ax_f.role_axes(ar.Channelout)[0].length, name='C', roles=[ar.Channel]),
        spatial_axis(ax_i, ax_f, padding['pad_d'], strides['str_d'], role=ar.Depth),
        spatial_axis(ax_i, ax_f, padding['pad_h'], strides['str_h'], role=ar.Height),
        spatial_axis(ax_i, ax_f, padding['pad_w'], strides['str_w'], role=ar.Width),
        ax.N
    ])

    inputs = ng.placeholder(axes=ax_i)
    filters = ng.placeholder(axes=ax_f)

    # randomly initialize
    input_value = rng.uniform(-1, 1, ax_i)
    filter_value = rng.uniform(-1, 1, ax_f)

    assert input_value.shape == ax_i.lengths
    assert filter_value.shape == ax_f.lengths

    inputs = ng.placeholder(ax_i)
    filters = ng.placeholder(ax_f)

    output = ng.convolution(conv_params, inputs, filters, axes=ax_o)
    targets = ng.placeholder(axes=output.axes)

    costs = ng.cross_entropy_binary(ng.sigmoid(output), targets)
    error = ng.sum(costs, out_axes=()) / ng.batch_size(costs)
    d_inputs = ng.deriv(error, inputs)
    d_filters = ng.deriv(error, filters)

    targets_value = rng.uniform(.1, 0.9, output.axes)

    conv_executor = executor([output, error, d_inputs, d_filters], inputs, filters, targets)
    result_ng, err_ng, gradI_ng, gradF_ng = conv_executor(input_value, filter_value, targets_value)

    # Now compute reference values via NEON
    NervanaObject.be.bsz = N
    neon_layer = Convolution(fshape=(R, S, K), padding=padding, strides=strides)

    inp = neon_layer.be.array(input_value.reshape(C * H * W * D, N))
    neon_layer.W = neon_layer.be.array(filter_value.reshape(C * R * S * T, K))
    neon_layer.dW = neon_layer.be.empty_like(neon_layer.W)
    neon_layer.configure((C, H, W))
    neon_layer.prev_layer = True
    neon_layer.allocate()
    neon_layer.set_deltas(DummyDeltaBuffers())

    result_ne = neon_layer.fprop(inp).get().reshape(output.axes.lengths)

    act_result_ne = 1. / (1.0 + np.exp(-result_ne))
    err = neon_layer.be.array((act_result_ne - targets_value).reshape(-1, N) / float(N))
    gradI_ne = neon_layer.bprop(err).get().reshape(ax_i.lengths)
    gradF_ne = neon_layer.dW.get().reshape(ax_f.lengths)

    # Compare fprop
    np.testing.assert_allclose(result_ng, result_ne, rtol=0, atol=1e-6)

    # Compare bprop
    np.testing.assert_allclose(gradI_ng, gradI_ne, rtol=0, atol=1e-6)

    # Compare update
    np.testing.assert_allclose(gradF_ng, gradF_ne, rtol=0, atol=1e-4)


@pytest.mark.xfail(strict=True)
def test_conv_flatten_deriv(transformer_factory):
    """
    Test deriv of conv followed by flatten
    """
    # set shape
    C, D, H, W, N = (3, 1, 28, 28, 8)
    C, T, R, S, K = (3, 1, 5, 5, 32)

    # i, f, o axes
    ax_i = ng.make_axes([ax.C, ax.D, ax.H, ax.W, ax.N])
    ax_f = ng.make_axes([ax.C, ax.T, ax.R, ax.S, ax.K])
    ax_o = ng.make_axes([
        ng.make_axis(32, roles=[ar.Channel]),
        ng.make_axis(1, roles=[ar.Depth]),
        ng.make_axis(24, roles=[ar.Height]),
        ng.make_axis(24, roles=[ar.Width]),
        ax.N
    ])
    ax_i.set_shape((C, D, H, W, N))
    ax_f.set_shape((C, T, R, S, K))
    params = dict(pad_d=0, pad_h=0, pad_w=0, str_d=1, str_h=1, str_w=1)
    axes_rsck = ng.make_axes([ax.R, ax.S, ax.C, ax.K])
    axes_rsck_prime = ng.make_axes([ng.make_axis(l) for l in axes_rsck.lengths])

    # broadcast input / filter axes
    image = ng.constant(np.ones(ax_i.lengths), ax_i)
    filter = ng.variable(axes_rsck_prime,
                         initial_value=np.ones((R, S, C, K)))
    filter_casted = ng.cast_axes(filter, axes_rsck)
    filter_casted = ng.expand_dims(filter_casted, ax.T, 0)
    filter_casted = ng.axes_with_order(filter_casted, axes=ax_f)

    # convolution
    output = ng.convolution(params, image, filter_casted, axes=ax_o)
    oC, oD, oH, oW, oN = output.axes
    output = ng.axes_with_order(output, axes=ng.make_axes([oN, oD, oH, oW, oC]))

    # slice away the oD
    out_slicing = [slice(None), 0, slice(None), slice(None), slice(None)]
    conv = ng.Slice(output, out_slicing)
    flatten = ng.flatten_at(conv, idx=1)

    # cost and grad
    cost = ng.sum(flatten, reduction_axes=flatten.axes)
    grad = ng.deriv(cost, filter)

    # compute
    conv_grad_comp = executor([conv, grad])
    conv_val, grad_val = conv_grad_comp()

    assert np.allclose(conv_val, np.zeros_like(conv_val) + 75.)
    assert np.allclose(grad_val, np.zeros_like(grad_val) + 4608.)
