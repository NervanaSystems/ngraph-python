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
from __future__ import print_function
import numpy as np
import ngraph as ng
from ngraph.testing.random import RandomTensorGenerator
import ngraph.transformers as ngt
from ngraph.op_graph.axes import spatial_axis

from ngraph.frontends.neon import ax, ar
from flexargparser import FlexNgraphArgparser


# This is currently unused
rng = RandomTensorGenerator(0, np.float32)


# Select a transformer
parser = FlexNgraphArgparser(description='simple conv example')
args = parser.parse_args()
transformer_name = args.backend

factory = ngt.make_transformer_factory(transformer_name)
ngt.set_transformer_factory(factory)

print("\n--------- conv example using ", transformer_name, "-----------\n")


def get_dimvals(inputs, filters):
    C, T, R, S, K = filters.shape
    _, D, H, W, N = inputs.shape
    P = H - R + 1
    Q = W - S + 1
    assert T == D == 1
    return C, T, R, S, K, D, H, W, N, P, Q


def np_fprop(inputs, filters):
    # Convolve inputs with filters
    C, T, R, S, K, D, H, W, N, P, Q = get_dimvals(inputs, filters)
    result = np.zeros((K, 1, P, Q, N), dtype=np.float32)
    for p, q in np.ndindex(P, Q):
        data = inputs[:, 0, p:p+R, q:q+S].reshape((C*R*S, N))
        result[:, 0, p, q] = np.dot(filters.reshape((C*R*S, K)).T, data)
    return result


def np_bprop(delta, filters, inputs):
    # Propagate gradients backwards
    C, T, R, S, K, D, H, W, N, P, Q = get_dimvals(inputs, filters)
    result = np.zeros(inputs.shape, dtype=np.float32)
    for p, q in np.ndindex(P, Q):
        data = delta[:, 0, p, q].reshape((K, N))
        result[:, 0:1, p:p+R, q:q+S] += np.dot(filters, data).reshape((C, 1, R, S, N))
    return result


def np_update(delta, filters, inputs):
    # Compute gradient of weights
    C, T, R, S, K, D, H, W, N, P, Q = get_dimvals(inputs, filters)
    result = np.zeros(filters.shape, dtype=np.float32)
    for p, q in np.ndindex(P, Q):
        data = delta[:, 0, p, q].reshape((K, N))
        result += np.dot(inputs[:, 0, p:p+R, q:q+S].reshape((C*R*S, N)), data.T).reshape(result.shape)
    return result


def test_conv():
    # This configuration has been tweaked to minimize overflow with 8.8 fixed point.
    N = 64  # 64 is the smallest possible for large N
    C, K = 16, 8  # C and K exchanged for f/bprop,  4 is the minimum that divides vec_size (or is it 8?)
    D, T = 1, 1
    H = W = 6  # for a 4x4 output feature map
    R = S = 3

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

    input_value = np.empty(ng.make_axes(ax_i).lengths, dtype=np.float32)
    filter_value = np.ones(ng.make_axes(ax_f).lengths, dtype=np.float32)
    input_value[:] = 0.5

    result_value = np_fprop(input_value, filter_value)
    assert input_value.shape == ax_i.lengths
    assert filter_value.shape == ax_f.lengths

    inputs = ng.placeholder(ax_i)
    filters = ng.placeholder(ax_f)

    output = ng.convolution(conv_params, inputs, filters, axes=ax_o)
    targets = ng.placeholder(axes=output.axes)

    diffs = output - targets
    costs = diffs * diffs
    error = ng.sum(costs, reduction_axes=costs.axes)
    d_inputs = ng.deriv(error, inputs)
    d_filters = ng.deriv(error, filters)

    # This makes all cost elements 0.0025
    targets_value = result_value - 0.05

    trafo = ngt.make_transformer()
    conv_executor = trafo.computation([output, costs, error, d_inputs, d_filters], inputs, filters, targets)
    result_ng, costs_ng, err_ng, gradI_ng, gradF_ng = conv_executor(input_value, filter_value, targets_value)

    diffs_value = result_value - targets_value
    costs_value = diffs_value * diffs_value
    # The derivative of costs with respect to the output
    delta_value = 2 * diffs_value
    gradI = np_bprop(delta_value, filter_value, input_value)
    gradF = np_update(delta_value, filter_value, input_value)
    print ("   fprop result", result_ng[:,0,3,3,63])
    print ("np fprop result", result_value[:,0,3,3,63])
    print ("   costs", costs_ng[:,0,3,3,63])
    print ("np costs", costs_value[:,0,3,3,63])
    print ("   error", err_ng)
    print ("np error", np.sum(costs_value))

    print ("   gradI", gradI_ng[:,0,3,3,63])
    print ("np gradI", gradI[:,0,3,3,63])
    print ("   gradF", gradF_ng[2,0,:,:,7])
    print ("np gradF", gradF[2,0,:,:,7])

    return trafo


transformer = test_conv()

# if transformer_name == 'flexgpu' and transformer.flex_manager.num_flex_tensors < 20:
#     print(transformer.flex_manager.stat_ids)
#     fm = transformer.flex_manager

#     print(fm.host_stats)
#     fm.transfer_stats()
#     print(fm.host_stats)
