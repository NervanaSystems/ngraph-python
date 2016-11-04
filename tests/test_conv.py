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
    C = 3
    K = 8
    D, T = 1, 1
    H = W = 32
    R = S = 2
    Transformer.make_transformer()
    padding = dict(pad_d=0, pad_h=0, pad_w=0)
    strides = dict(str_d=1, str_h=1, str_w=1)
    conv_params = padding.copy()
    conv_params.update(strides)
    Nx = ng.make_axis(N, batch=True)

    Cx = ng.make_axis(C, name='C')
    Dx = ng.make_axis(D, name='D')
    Hx = ng.make_axis(H, name='H')
    Wx = ng.make_axis(W, name='W')
    Tx = ng.make_axis(T, name='T')
    Rx = ng.make_axis(R, name='R')
    Sx = ng.make_axis(S, name='S')
    Kx = ng.make_axis(K, name='K')
    dtypeu = np.float32

    inputs = ng.placeholder(axes=ng.make_axes([Cx, Dx, Hx, Wx, Nx]))
    filters = ng.placeholder(axes=ng.make_axes([Cx, Tx, Rx, Sx, Kx]))

    # randomly initialize
    input_value = rng.uniform(-1, 1, inputs.axes)
    filter_value = rng.uniform(-1, 1, filters.axes)

    assert input_value.shape == tuple([ax.length for ax in [Cx, Dx, Hx, Wx, Nx]])
    assert filter_value.shape == tuple([ax.length for ax in [Cx, Tx, Rx, Sx, Kx]])

    # compute convolution with graph
    output = ng.convolution(conv_params, inputs, filters)
    targets = ng.placeholder(axes=output.axes)

    costs = ng.cross_entropy_binary(ng.sigmoid(output), targets)
    error = ng.sum(costs, out_axes=()) / ng.batch_size(costs)
    d_inputs = ng.deriv(error, inputs)
    d_filters = ng.deriv(error, filters)

    targets_value = rng.uniform(.1, 0.9, output.axes)

    conv_executor = executor([output, error, d_inputs, d_filters], inputs, filters, targets)
    result_ng, err_ng, gradI_ng, gradF_ng = conv_executor(input_value, filter_value, targets_value)


    #### Now compute reference values via NEON
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
    gradI_ne = neon_layer.bprop(err).get().reshape(inputs.axes.lengths)
    gradF_ne = neon_layer.dW.get().reshape(filters.axes.lengths)

    # Compare fprop
    np.testing.assert_allclose(result_ng, result_ne, rtol=0, atol=1e-6)

    # Compare bprop
    np.testing.assert_allclose(gradI_ng, gradI_ne, rtol=0, atol=1e-6)

    # Compare update
    np.testing.assert_allclose(gradF_ng, gradF_ne, rtol=0, atol=1e-6)


