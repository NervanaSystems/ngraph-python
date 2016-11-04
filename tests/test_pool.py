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
from neon.layers.layer import Pooling

rng = RandomTensorGenerator(0, np.float32)

NervanaObject.be = gen_backend()


class DummyDeltaBuffers(object):
    """
    Dummy class for delta buffers needed by neon
    """
    def __init__(self):
        self.buffers = [None]


def test_pooling():
    """
    test pooling forward and backward path
    """
    N = 128
    C = 3
    D = 1
    H = W = 32

    J = T = 1
    R = S = 2
    Transformer.make_transformer()

    padding = dict(pad_d=0, pad_h=0, pad_w=0, pad_c=0)
    strides = dict(str_d=1, str_h=1, str_w=1, str_c=1)
    fshape = dict(J=J, T=T, R=R, S=S)

    pool_params = dict(op='max')
    pool_params.update(padding)
    pool_params.update(strides)
    pool_params.update(fshape)

    Nx = ng.Axis(N, batch=True)

    Cx = ng.Axis(C, name='C')
    Dx = ng.Axis(D, name='D')
    Hx = ng.Axis(H, name='H')
    Wx = ng.Axis(W, name='W')

    inputs = ng.placeholder(axes=ng.Axes([Cx, Dx, Hx, Wx, Nx]))

    # randomly initialize
    input_value = rng.uniform(-1, 1, inputs.axes)

    assert input_value.shape == tuple([ax.length for ax in [Cx, Dx, Hx, Wx, Nx]])

    # compute convolution with graph
    output = ng.pooling(pool_params, inputs)
    targets = ng.placeholder(axes=output.axes)

    costs = ng.cross_entropy_binary(ng.sigmoid(output), targets)
    error = ng.sum(costs, out_axes=()) / ng.batch_size(costs)
    d_inputs = ng.deriv(error, inputs)

    targets_value = rng.uniform(.1, 0.9, output.axes)

    conv_executor = executor([output, error, d_inputs], inputs, targets)
    result_ng, err_ng, gradI_ng = conv_executor(input_value, targets_value)

    # Now compute reference values via NEON
    NervanaObject.be.bsz = N
    neon_layer = Pooling(fshape=fshape, padding=padding, strides=strides, op="max")

    inp = neon_layer.be.array(input_value.reshape(C * H * W * D, N))
    neon_layer.configure((C, H, W))
    neon_layer.prev_layer = True
    neon_layer.allocate()
    neon_layer.set_deltas(DummyDeltaBuffers())

    result_ne = neon_layer.fprop(inp).get().reshape(output.axes.lengths)

    act_result_ne = 1. / (1.0 + np.exp(-result_ne))
    err = neon_layer.be.array((act_result_ne - targets_value).reshape(-1, N) / float(N))
    gradI_ne = neon_layer.bprop(err).get().reshape(inputs.axes.lengths)

    # Compare fprop
    np.testing.assert_allclose(result_ng, result_ne, rtol=0, atol=1e-6)

    # Compare bprop
    np.testing.assert_allclose(gradI_ng, gradI_ne, rtol=0, atol=1e-6)
