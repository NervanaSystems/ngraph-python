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
from conv_ref import ConvLayerRef

from neon import NervanaObject
from neon.backends import gen_backend
from neon.layers.layer import Convolution

rng = RandomTensorGenerator(0, np.float32)

NervanaObject.be = gen_backend()

def test_convolution_fprop():
    """
    test convolution forward path
    """
    N = 1
    C = 3
    K = 8
    D, T = 1, 1
    H = W = 32
    R = S = 2
    Transformer.make_transformer()
    padding = dict(pad_d=0, pad_h=0, pad_w=0)
    strides = dict(str_d=1, str_h=1, str_w=1)
    dims = padding.copy()
    dims.update(strides)
    Nx = ng.Axis(N, batch=True)

    Cx = ng.Axis(C, name='C')
    Dx = ng.Axis(D, name='D')
    Hx = ng.Axis(H, name='H')
    Wx = ng.Axis(W, name='W')
    Tx = ng.Axis(T, name='T')
    Rx = ng.Axis(R, name='R')
    Sx = ng.Axis(S, name='S')
    Kx = ng.Axis(K, name='K')
    dtypeu = np.float32

    inputs = ng.placeholder(axes=ng.Axes([Cx, Dx, Hx, Wx, Nx]))
    filters = ng.placeholder(axes=ng.Axes([Cx, Tx, Rx, Sx, Kx]))

    # randomly initialize
    input_value = rng.uniform(-1, 1, inputs.axes)
    filter_value = rng.uniform(-1, 1, filters.axes)

    assert input_value.shape == tuple([ax.length for ax in [Cx, Dx, Hx, Wx, Nx]])
    assert filter_value.shape == tuple([ax.length for ax in [Cx, Tx, Rx, Sx, Kx]])

    # compute convolution with graph
    output = ng.convolution(dims, inputs, filters)
    d_filters = ng.deriv(output, filters)
    d_inputs = ng.deriv(output, inputs)

    result_og = executor(output, inputs, filters)(input_value, filter_value)


    NervanaObject.be.bsz = N
    neon_layer = Convolution(fshape=(R, S, K), padding=padding, strides=strides)

    inp = neon_layer.be.array(input_value.reshape(C * H * W * D, N))
    neon_layer.W = neon_layer.be.array(filter_value.reshape(C * R * S * T, K))
    neon_layer.configure((C, H, W))
    neon_layer.prev_layer = True
    neon_layer.allocate()

    result_value = neon_layer.fprop(inp).get().reshape(output.axes.lengths)

    # neon_layer.allocate_deltas(deltas_buffer)
    # deltas_buffer.allocate_buffers()
    # neon_layer.set_deltas(deltas_buffer)



    # ref_layer = ConvLayerRef(Nx.length,
    #                          C, (H, W), (R, S), K,
    #                          dtypeu,
    #                          strides=dims['str_h'],
    #                          padding=dims['pad_h'])
    # ref_layer.weights = filter_value.reshape(C * R * S, K).T.astype(dtypeu)
    # ref_layer.fprop(input_value.reshape(C * H * W, N).T.astype(dtypeu))
    # result_value = ref_layer.y.copy().T.reshape(output.axes.lengths)

    # result_value = np.zeros(output.axes.lengths, dtype=np.float32)

    # np_convolution(input_value, filter_value, result_value)
    np.testing.assert_allclose(result_og, result_value, rtol=1e-3, atol=1e-6)
