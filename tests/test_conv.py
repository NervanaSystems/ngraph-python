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

# import numpy as np
# import pytest
# from neon import NervanaObject
# from neon.backends import gen_backend
# from neon.layers.layer import Convolution

# import ngraph as ng
# from ngraph.frontends.neon import ax, ar
# from ngraph.frontends.neon.layer import output_dim
# from ngraph.testing import ExecutorFactory, RandomTensorGenerator, executor

# rng = RandomTensorGenerator(0, np.float32)


# NervanaObject.be = gen_backend()

# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
import itertools as itt
from ngraph.op_graph.convolution import bprop_conv, update_conv
from ngraph.testing import ExecutorFactory, RandomTensorGenerator, executor
from ngraph.frontends.neon.layer import output_dim


rng = RandomTensorGenerator(0, np.float32)


def slicable(dim, pad=0):
    """
    colapse outer dimensions into one and preserve inner dimension
    this allows for easy cpu convolution in numpy

    Arguments:
        dim (tuple): dimensions list in a tuple
        pad (int):  how many pixel paddings
    """
    dim0 = np.prod(dim[:-1]) + pad
    return (dim0, dim[-1])


def pixel_indices(T, R, S, D, H, W, C, mt, pr, qs):
    HW = H * W
    DHW = D * H * W
    imax = C * DHW

    idx = []
    for c, t, r, s in itt.product(range(C), range(T), range(R), range(S)):

        ci = c * DHW

        z = mt + t
        zi = ci + z * HW
        zb = z >= 0 and z < D

        y = pr + r
        yi = zi + y * W
        yb = zb and y >= 0 and y < H

        x = qs + s

        if yb and x >= 0 and x < W:
            xi = yi + x
        else:
            xi = imax  # out of bounds

        idx.append(xi)

    return idx


def reference_conv(dimI, dimF, dimO, conv_params, valI, valF, valE):
    (K, M, P, Q, N) = dimO
    (C, D, H, W, N) = dimI
    (C, T, R, S, K) = dimF
    pad_d, pad_h, pad_w = conv_params['pad_d'], conv_params['pad_h'], conv_params['pad_w']
    str_d, str_h, str_w = conv_params['str_d'], conv_params['str_h'], conv_params['str_w']
    dtype = np.float32

    no_pad_I = slicable(dimI)
    cpuI = np.zeros(slicable(dimI, 1), dtype=dtype)
    cpuI[:no_pad_I[0], :] = valI.reshape(no_pad_I)

    cpuF = valF.reshape(slicable(dimF))
    cpuE = valE

    # ======numpy===========
    # cpu output arrays
    cpuO = np.zeros(dimO, dtype=dtype)
    cpuB = np.zeros(slicable(dimI, 1), dtype=dtype)
    cpuU = np.zeros(slicable(dimF), dtype=dtype)

    for m, p, q in itt.product(range(M), range(P), range(Q)):
        mt = m * str_d - pad_d
        pr = p * str_h - pad_h
        qs = q * str_w - pad_w

        idx = pixel_indices(T, R, S, D, H, W, C, mt, pr, qs)

        cpuO[:, m, p, q, :] = np.dot(cpuF.T, cpuI[idx, :])

        cpuB[idx, :] += np.dot(cpuF, cpuE[:, m, p, q, :])

        cpuU += np.dot(cpuI[idx, :], cpuE[:, m, p, q, :].T)

    outB = cpuB[:-1, :].reshape(dimI)
    outU = cpuU.reshape(dimF)
    return (cpuO, outB, outU)


class ConvParams(object):
    def __init__(self, C=1, N=1, K=1, D=1, H=1, W=1, T=1, R=1, S=1,
                 pad_d=0, pad_h=0, pad_w=0,
                 str_d=1, str_h=1, str_w=1):

        M = output_dim(D, T, pad_d, str_d)
        P = output_dim(H, R, pad_h, str_h)
        Q = output_dim(W, S, pad_w, str_w)

        self.dimO = (K, M, P, Q, N)
        self.dimI = (C, D, H, W, N)
        self.dimF = (C, T, R, S, K)

        self.conv_params = dict(
            pad_d=pad_d, pad_h=pad_h, pad_w=pad_w,
            str_d=str_d, str_h=str_h, str_w=str_w,
            dil_d=1, dil_h=1, dil_w=1
        )

        batch_axis = ng.make_axis(name='N', length=N)

        self.ax_i = ng.make_axes([
            ng.make_axis(name='C', length=C),
            ng.make_axis(name='D', length=D),
            ng.make_axis(name='H', length=H),
            ng.make_axis(name='W', length=W),
            batch_axis
        ])

        self.ax_f = ng.make_axes([
            ng.make_axis(name='C', length=C),
            ng.make_axis(name='D', length=T),
            ng.make_axis(name='H', length=R),
            ng.make_axis(name='W', length=S),
            ng.make_axis(name='K', length=K),
        ])

        self.ax_o = ng.make_axes([
            ng.make_axis(name='C', length=K),
            ng.make_axis(name='D', length=M),
            ng.make_axis(name='H', length=P),
            ng.make_axis(name='W', length=Q),
            batch_axis
        ])


@pytest.fixture()
def n64_hw32_c32_3x3():
    return dict(C=32, N=64, K=32, H=32, W=32, R=3, S=3)


@pytest.fixture()
def n128_hw32_c3_2x2():
    return dict(C=3, N=128, K=2, H=32, W=32, R=2, S=2)


@pytest.fixture()
def n4_hw16_c3_5x5():
    return dict(C=3, N=4, K=8, H=16, W=16, R=5, S=5)


def test_conv(transformer_factory, n64_hw32_c32_3x3):
    cf = ConvParams(**n64_hw32_c32_3x3)

    inputs = ng.placeholder(axes=cf.ax_i)
    filters = ng.placeholder(axes=cf.ax_f)

    # randomly initialize
    input_value = rng.uniform(-0.5, 0.5, cf.ax_i)
    filter_value = rng.uniform(-0.5, 0.5, cf.ax_f)
    error_value = rng.uniform(-0.5, 0.5, cf.ax_o)

    inputs = ng.placeholder(cf.ax_i)
    filters = ng.placeholder(cf.ax_f)
    errors = ng.placeholder(cf.ax_o)

    output = ng.convolution(cf.conv_params, inputs, filters, axes=cf.ax_o)
    bprop_out = bprop_conv(errors, inputs, filters, output)
    updat_out = update_conv(errors, inputs, filters, output)

    with executor([output, bprop_out, updat_out], inputs, filters, errors) as conv_executor:
        result_ng, gradI_ng, gradF_ng = conv_executor(input_value, filter_value, error_value)

    # Compute reference with NumPy
    result_np, gradI_np, gradF_np = reference_conv(cf.dimI, cf.dimF, cf.dimO,
                                                   cf.conv_params,
                                                   input_value, filter_value, error_value)

    # Compare fprop
    assert np.allclose(result_ng, result_np, rtol=0, atol=0.5)

    # Compare bprop
    assert np.allclose(gradI_ng, gradI_np, rtol=0, atol=0.5)

    # Compare update
    assert np.allclose(gradF_ng, gradF_np, rtol=0, atol=2)


def test_wrong_filters_shape_length():
    """
    test wrong filters shape length
    """
    cf = ConvParams()
    ax_f = cf.ax_f[:-1]

    inputs = ng.placeholder(cf.ax_i)
    filters = ng.placeholder(ax_f)

    with pytest.raises(ValueError) as exinfo:
        ng.convolution(cf.conv_params, inputs, filters, {})
    assert str(exinfo.value) == 'convolution filter shape must be length 5, found {}'\
        .format(len(ax_f))


def test_wrong_input_shape_length():
    """
    test wrong input shape length
    """
    cf = ConvParams()
    ax_i = cf.ax_i[:-1]

    inputs = ng.placeholder(ax_i)
    filters = ng.placeholder(cf.ax_f)

    with pytest.raises(ValueError) as exinfo:
        ng.convolution(cf.conv_params, inputs, filters, {})
    assert str(exinfo.value) == 'convolution input shape must be length 5, found {}'\
        .format(len(ax_i))


def test_first_axes_not_same():
    """
    test first axes are not the same
    """
    cf = ConvParams()
    ax_i = cf.ax_i[1:2] + cf.ax_i[0:1] + cf.ax_i[2:]  # D, C, H, W, N

    inputs = ng.placeholder(ax_i)
    filters = ng.placeholder(cf.ax_f)

    with pytest.raises(ValueError) as exinfo:
        ng.convolution(cf.conv_params, inputs, filters, {})
    assert str(exinfo.value) == 'the first axis in input {inputs} and filter {filters} ' \
        'are not the same.'.format(
            inputs=inputs.axes[0],
            filters=filters.axes[0])


def test_convolution_backprop(transformer_factory, n128_hw32_c3_2x2):
    """
    test convolution backprop path
    """
    cf = ConvParams(**n128_hw32_c3_2x2)
    inputs = ng.placeholder(axes=cf.ax_i)
    filters = ng.placeholder(axes=cf.ax_f)

    # randomly initialize
    input_value = rng.uniform(-1, 1, cf.ax_i)
    filter_value = rng.uniform(-1, 1, cf.ax_f)

    output = ng.sum(ng.convolution(cf.conv_params, inputs, filters, cf.ax_o), out_axes=())

    with ExecutorFactory() as factory:
        dcdf_sym_fun = factory.derivative(output, filters, inputs)
        dcdf_num_fun = factory.numeric_derivative(output, filters, .01, inputs)
        dcdf_sym_val = dcdf_sym_fun(filter_value, input_value)
        dcdf_num_val = dcdf_num_fun(filter_value, input_value)

        ng.testing.assert_allclose(dcdf_sym_val, dcdf_num_val, rtol=1)


def test_conv_flatten_deriv(transformer_factory, n4_hw16_c3_5x5):
    """
    Test deriv of conv followed by flatten
    """
    cf = ConvParams(**n4_hw16_c3_5x5)

    axes_rsck = ng.make_axes([cf.ax_f[2], cf.ax_f[3], cf.ax_f[0], cf.ax_f[-1]])
    axes_rsck_prime = ng.make_axes([ng.make_axis(name=ax.name + 'p', length=ax.length)
                                    for ax in axes_rsck])
    axes_nmpqk = ng.make_axes([cf.ax_o[-1], cf.ax_o[1], cf.ax_o[2], cf.ax_o[3], cf.ax_o[0]])

    # broadcast input / filter axes
    input_var = ng.variable(cf.ax_i)
    input_var.input = True
    input_val = np.ones(input_var.axes.lengths)

    filter_rsck_prime = ng.variable(axes_rsck_prime)
    filter_var = filter_rsck_prime
    filter_rsck = ng.cast_axes(filter_rsck_prime, axes_rsck)
    filter_trsck = ng.expand_dims(filter_rsck, cf.ax_f[1], 0)
    filter_ctrsk = ng.axes_with_order(filter_trsck, axes=cf.ax_f)

    # convolution
    output_kmpqn = ng.convolution(cf.conv_params, input_var, filter_ctrsk, axes=cf.ax_o)
    output_nmpqk = ng.axes_with_order(output_kmpqn, axes=axes_nmpqk)

    # slice away the oD
    out_slicing = [slice(None), 0, slice(None), slice(None), slice(None)]
    output_npqk = ng.tensor_slice(output_nmpqk, out_slicing)

    output = ng.flatten_at(output_npqk, idx=1)

    # cost and grad
    cost = ng.sum(output, out_axes=())

    filter_var.input = True
    filter_val = np.ones(filter_var.axes.lengths)

    with ExecutorFactory() as factory:

        conv_comp = factory.executor(output, filter_var, input_var)
        grad_filter_num_comp = factory.numeric_derivative(cost, filter_var, 1.0, input_var)
        grad_filter_sym_comp = factory.derivative(cost, filter_var, input_var)

        grad_input_num_comp = factory.numeric_derivative(cost, input_var, 1.0, filter_var)
        grad_input_sym_comp = factory.derivative(cost, input_var, filter_var)

        conv_val = conv_comp(filter_val, input_val)
        conv_val_num = np.empty_like(conv_val)
        conv_val_num.fill(np.prod(cf.ax_f.lengths[:-1]))
        assert ng.testing.allclose(conv_val, conv_val_num)

        grad_filter_num_val = grad_filter_num_comp(filter_val, input_val)
        grad_filter_sym_val = grad_filter_sym_comp(filter_val, input_val)
        assert ng.testing.allclose(grad_filter_num_val, grad_filter_sym_val)

        grad_input_num_val = grad_input_num_comp(input_val, filter_val)
        grad_input_sym_val = grad_input_sym_comp(input_val, filter_val)
        assert ng.testing.allclose(grad_input_num_val, grad_input_sym_val)
