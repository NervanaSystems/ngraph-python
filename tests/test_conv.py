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
from neon import NervanaObject
from neon.backends import gen_backend
from neon.layers.layer import Convolution

import ngraph as ng
from ngraph.frontends.neon import ax, ar
from ngraph.frontends.neon.layer import output_dim
from ngraph.testing import ExecutorFactory, RandomTensorGenerator, executor

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
    dilation = dict(dil_d=1, dil_h=1, dil_w=1)
    conv_params = padding.copy()
    conv_params.update(strides)
    conv_params.update(dilation)

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
    dilation = dict(dil_d=1, dil_h=1, dil_w=1)
    conv_params = padding.copy()
    conv_params.update(strides)
    conv_params.update(dilation)

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
    dilation = dict(dil_d=1, dil_h=1, dil_w=1)
    conv_params = padding.copy()
    conv_params.update(strides)
    conv_params.update(dilation)

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
    dilation = dict(dil_d=1, dil_h=1, dil_w=1)
    conv_params = padding.copy()
    conv_params.update(strides)
    conv_params.update(dilation)

    C = 3
    D = 1
    ax_C = ng.make_axis(length=C, batch=True).named('C')
    ax_D = ng.make_axis(length=D, batch=True).named('D')

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


def test_convolution_backprop(transformer_factory):
    """
    test convolution backprop path
    """
    N = 128
    C, K = 3, 2
    D, T = 1, 1
    H = W = 32
    R = S = 2

    padding = dict(pad_d=0, pad_h=0, pad_w=0)
    strides = dict(str_d=1, str_h=1, str_w=1)
    dilation = dict(dil_d=1, dil_h=1, dil_w=1)
    conv_params = padding.copy()
    conv_params.update(strides)
    conv_params.update(dilation)

    ax_i = ng.make_axes([ax.C, ax.D, ax.H, ax.W, ax.N])
    ax_f = ng.make_axes([ax.C, ax.T, ax.R, ax.S, ax.K])
    ax_i.set_shape((C, D, H, W, N))
    ax_f.set_shape((C, T, R, S, K))
    ax_o = ng.make_axes([
        ng.make_axis(roles=[ar.features_input]).named('C'),
        ng.make_axis(roles=[ar.features_0]).named('D'),
        ng.make_axis(roles=[ar.features_1]).named('H'),
        ng.make_axis(roles=[ar.features_2]).named('W'),
        ax.N
    ])

    ax_o[:-1].set_shape((
        K,
        output_dim(D, T, padding['pad_d'], strides['str_d']),
        output_dim(H, R, padding['pad_h'], strides['str_h']),
        output_dim(W, S, padding['pad_w'], strides['str_w']))
    )

    inputs = ng.placeholder(axes=ax_i)
    filters = ng.placeholder(axes=ax_f)

    # randomly initialize
    input_value = rng.uniform(-1, 1, ax_i)
    filter_value = rng.uniform(-1, 1, ax_f)

    assert input_value.shape == ax_i.lengths
    assert filter_value.shape == ax_f.lengths

    output = ng.sum(ng.convolution(conv_params, inputs, filters, ax_o), out_axes=())

    with ExecutorFactory() as factory:
        dcdf_sym_fun = factory.derivative(output, filters, inputs)
        dcdf_num_fun = factory.numeric_derivative(output, filters, .01, inputs)
        dcdf_sym_val = dcdf_sym_fun(filter_value, input_value)
        dcdf_num_val = dcdf_num_fun(filter_value, input_value)

        ng.testing.assert_allclose(dcdf_sym_val, dcdf_num_val, rtol=1)


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
    dilation = dict(dil_d=1, dil_h=1, dil_w=1)
    conv_params = padding.copy()
    conv_params.update(strides)
    conv_params.update(dilation)

    ax_i = ng.make_axes([ax.C, ax.D, ax.H, ax.W, ax.N])
    ax_f = ng.make_axes([ax.C, ax.T, ax.R, ax.S, ax.K])
    ax_i.set_shape((C, D, H, W, N))
    ax_f.set_shape((C, T, R, S, K))

    ax_o = ng.make_axes([
        ng.make_axis(roles=[ar.features_input]).named('C'),
        ng.make_axis(roles=[ar.features_0]).named('D'),
        ng.make_axis(roles=[ar.features_1]).named('H'),
        ng.make_axis(roles=[ar.features_2]).named('W'),
        ax.N
    ])

    ax_o[:-1].set_shape((
        K,
        output_dim(D, T, padding['pad_d'], strides['str_d']),
        output_dim(H, R, padding['pad_h'], strides['str_h']),
        output_dim(W, S, padding['pad_w'], strides['str_w']))
    )

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

    with executor([output, error, d_inputs, d_filters], inputs, filters, targets) as conv_executor:
        result_ng, err_ng, gradI_ng, gradF_ng = \
            conv_executor(input_value, filter_value, targets_value)

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
    ng.testing.assert_allclose(result_ng, result_ne, rtol=0, atol=1e-6)

    # Compare bprop
    ng.testing.assert_allclose(gradI_ng, gradI_ne, rtol=0, atol=1e-6)

    # Compare update
    ng.testing.assert_allclose(gradF_ng, gradF_ne, rtol=0, atol=1e-4)


def test_conv_flatten_deriv(transformer_factory):
    """
    Test deriv of conv followed by flatten
    """

    # set shape
    # NOTE: N must be >= 4 for GPU, but for CPU this could be decreased to
    # speed up the test
    N = 4
    C, D, H, W = (3, 1, 28, 28)
    T, R, S, K = (1, 5, 5, 8)

    params = dict(pad_d=0, pad_h=0, pad_w=0, str_d=1, str_h=1, str_w=1, dil_d=1, dil_h=1, dil_w=1)

    # i, f, o axes
    ax_i = ng.make_axes([ax.C, ax.D, ax.H, ax.W, ax.N])
    ax_f = ng.make_axes([ax.C, ax.T, ax.R, ax.S, ax.K])
    ax_o = ng.make_axes([
        ng.make_axis(roles=[ar.features_input]).named('C'),
        ng.make_axis(roles=[ar.features_0]).named('D'),
        ng.make_axis(roles=[ar.features_1]).named('H'),
        ng.make_axis(roles=[ar.features_2]).named('W'),
        ax.N
    ])

    ax_i.set_shape((C, D, H, W, N))
    ax_f.set_shape((C, T, R, S, K))
    ax_o.set_shape((K, D - T + 1, H - R + 1, W - S + 1, N))
    axes_rsck = ng.make_axes([ax.R, ax.S, ax.C, ax.K])
    axes_rsck_prime = ng.make_axes([ng.make_axis(axis.length).named(axis.name + 'p')
                                    for axis in axes_rsck])
    axes_nmpqk = ng.make_axes([ax_o[-1], ax_o[1], ax_o[2], ax_o[3], ax_o[0]])

    # broadcast input / filter axes
    input_var = ng.variable(ax_i).named('input')
    input_var.input = True
    input_val = np.ones(input_var.axes.lengths)

    filter_rsck_prime = ng.variable(axes_rsck_prime)
    filter_var = filter_rsck_prime
    filter_rsck = ng.cast_axes(filter_rsck_prime, axes_rsck)
    filter_trsck = ng.expand_dims(filter_rsck, ax.T, 0)
    filter_ctrsk = ng.axes_with_order(filter_trsck, axes=ax_f)

    # convolution
    output_kmpqn = ng.convolution(params, input_var, filter_ctrsk, axes=ax_o)
    output_nmpqk = ng.axes_with_order(output_kmpqn, axes=axes_nmpqk)

    # slice away the oD
    out_slicing = [slice(None), 0, slice(None), slice(None), slice(None)]
    output_npqk = ng.tensor_slice(output_nmpqk, out_slicing)

    output = ng.flatten_at(output_npqk, idx=1)

    # cost and grad
    cost = ng.sum(output, out_axes=())

    filter_var.input = True
    filter_var.named('filter')
    filter_val = np.ones(filter_var.axes.lengths)

    with ExecutorFactory() as factory:

        conv_comp = factory.executor(output, filter_var, input_var)
        grad_filter_num_comp = factory.numeric_derivative(cost, filter_var, 1.0, input_var)
        grad_filter_sym_comp = factory.derivative(cost, filter_var, input_var)

        grad_input_num_comp = factory.numeric_derivative(cost, input_var, 1.0, filter_var)
        grad_input_sym_comp = factory.derivative(cost, input_var, filter_var)

        conv_val = conv_comp(filter_val, input_val)
        conv_val_num = np.empty_like(conv_val)
        conv_val_num.fill(C * T * R * S)
        assert ng.testing.allclose(conv_val, conv_val_num)

        grad_filter_num_val = grad_filter_num_comp(filter_val, input_val)
        grad_filter_sym_val = grad_filter_sym_comp(filter_val, input_val)
        assert ng.testing.allclose(grad_filter_num_val, grad_filter_sym_val)

        grad_input_num_val = grad_input_num_comp(input_val, filter_val)
        grad_input_sym_val = grad_input_sym_comp(input_val, filter_val)
        assert ng.testing.allclose(grad_input_num_val, grad_input_sym_val)
