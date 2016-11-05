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

import ngraph as ng
from ngraph.util.utils import executor
from ngraph.util.utils import RandomTensorGenerator
from ngraph.transformers import Transformer
from ngraph.op_graph.axes import spatial_axis
from ngraph.frontends.neon import ax, ar
from neon import NervanaObject
from neon.backends import gen_backend
from neon.layers.layer import Convolution

rng = RandomTensorGenerator(0, np.float32)

NervanaObject.be = gen_backend()


class DummyDeltaBuffers(object):
    """
    Dummy class for delta buffers needed by neon
    """
    def __init__(self):
        self.buffers = [None]


def test_convolution():
    """
    test convolution forward path
    """
    N = 128
    C, K = 3, 8
    D, T = 1, 1
    H = W = 32
    R = S = 2
    Transformer.make_transformer()
    padding = dict(pad_d=0, pad_h=0, pad_w=0)
    strides = dict(str_d=1, str_h=1, str_w=1)
    conv_params = padding.copy()
    conv_params.update(strides)

    ax.N.length = N
    ax.C.length = C
    ax.D.length = D
    ax.H.length = H
    ax.W.length = W
    ax.T.length = T
    ax.R.length = R
    ax.S.length = S
    ax.K.length = K

    ax_i = ng.make_axes([ax.C, ax.D, ax.H, ax.W, ax.N])
    ax_f = ng.make_axes([ax.C, ax.T, ax.R, ax.S, ax.K])

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

    # compute convolution with graph
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
    np.testing.assert_allclose(gradF_ng, gradF_ne, rtol=0, atol=1e-6)
