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
from ngraph.op_graph.convolution import bprop_conv, update_conv
from ngraph.testing import RandomTensorGenerator, executor
from ngraph.frontends.neon.layer import output_dim
from ngraph.frontends.neon import ax

pytestmark = [pytest.mark.transformer_dependent,
              pytest.mark.flex_only]

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
    for c in range(C):
        ci = c * DHW

        for t in range(T):
            z = mt + t
            zi = ci + z * HW
            zb = z >= 0 and z < D

            for r in range(R):
                y = pr + r
                yi = zi + y * W
                yb = zb and y >= 0 and y < H

                for s in range(S):
                    x = qs + s
                    if yb and x >= 0 and x < W:
                        xi = yi + x
                    else:
                        xi = imax  # out of bounds

                    idx.append(xi)
    return idx


def reference_conv(C, N, K, D, H, W, T, R, S, M, P, Q,
                   pad_d, pad_h, pad_w, str_d, str_h, str_w,
                   valI, valF, valE):
    dimO = (K, M, P, Q, N)
    dimI = (C, D, H, W, N)
    dimF = (C, T, R, S, K)
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

    for m in range(M):
        mt = m * str_d - pad_d

        for p in range(P):
            pr = p * str_h - pad_h

            for q in range(Q):
                qs = q * str_w - pad_w

                idx = pixel_indices(T, R, S, D, H, W, C, mt, pr, qs)

                cpuO[:, m, p, q, :] = np.dot(cpuF.T, cpuI[idx, :])

                cpuB[idx, :] += np.dot(cpuF, cpuE[:, m, p, q, :])

                cpuU += np.dot(cpuI[idx, :], cpuE[:, m, p, q, :].T)

    outB = cpuB[:-1, :].reshape(dimI)
    outU = cpuU.reshape(dimF)
    return (cpuO, outB, outU)


def test_conv(transformer_factory):
    """
    TODO: make this more interesting
    """
    N, C, K = 64, 32, 32
    D, H, W = 1, 32, 32
    T, R, S = 1, 3, 3

    pad_d, pad_h, pad_w = 0, 0, 0
    str_d, str_h, str_w = 1, 1, 1
    dil_d, dil_h, dil_w = 1, 1, 1

    M = output_dim(D, T, pad_d, str_d)
    P = output_dim(H, R, pad_h, str_h)
    Q = output_dim(W, S, pad_w, str_w)

    padding = dict(pad_d=pad_d, pad_h=pad_h, pad_w=pad_w)
    strides = dict(str_d=str_d, str_h=str_h, str_w=str_w)
    dilation = dict(dil_d=dil_d, dil_h=dil_h, dil_w=dil_w)
    conv_params = padding.copy()
    conv_params.update(strides)
    conv_params.update(dilation)

    ax_i = ng.make_axes([
        ng.make_axis(name='C'),
        ng.make_axis(name='D'),
        ng.make_axis(name='H'),
        ng.make_axis(name='W'),
        ax.N
    ])

    ax_f = ng.make_axes([
        ng.make_axis(name='C'),
        ng.make_axis(name='D'),
        ng.make_axis(name='H'),
        ng.make_axis(name='W'),
        ng.make_axis(name='K'),
    ])

    ax_o = ng.make_axes([
        ng.make_axis(name='C'),
        ng.make_axis(name='D'),
        ng.make_axis(name='H'),
        ng.make_axis(name='W'),
        ax.N
    ])

    ax_i.set_shape((C, D, H, W, N))
    ax_f.set_shape((C, T, R, S, K))
    ax_o[:-1].set_shape((K, M, P, Q))

    inputs = ng.placeholder(axes=ax_i)
    filters = ng.placeholder(axes=ax_f)

    # randomly initialize
    input_value = rng.uniform(-0.5, 0.5, ax_i)
    filter_value = rng.uniform(-0.5, 0.5, ax_f)
    error_value = rng.uniform(-0.5, 0.5, ax_o)

    assert input_value.shape == ax_i.lengths
    assert filter_value.shape == ax_f.lengths

    inputs = ng.placeholder(ax_i)
    filters = ng.placeholder(ax_f)
    errors = ng.placeholder(ax_o)

    output = ng.convolution(conv_params, inputs, filters, axes=ax_o)
    bprop_out = bprop_conv(errors, inputs, filters, output)
    updat_out = update_conv(errors, inputs, filters, output)

    with executor([output, bprop_out, updat_out], inputs, filters, errors) as conv_executor:
        result_ng, gradI_ng, gradF_ng = conv_executor(input_value, filter_value, error_value)

    # Compute reference with NumPy
    result_np, gradI_np, gradF_np = reference_conv(C, N, K, D, H, W, T, R, S, M, P, Q,
                                                   pad_d, pad_h, pad_w, str_d, str_h, str_w,
                                                   input_value, filter_value, error_value)

    # Compare fprop
    assert np.allclose(result_ng, result_np, rtol=0, atol=0.5)

    # Compare bprop
    assert np.allclose(gradI_ng, gradI_np, rtol=0, atol=0.5)

    # Compare update
    assert np.allclose(gradF_ng, gradF_np, rtol=0, atol=2)
