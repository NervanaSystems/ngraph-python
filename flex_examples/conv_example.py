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
from ngraph.frontends.neon import NgraphArgparser


rng = RandomTensorGenerator(0, np.float32)


# Select a transformer
parser = NgraphArgparser(description='simple conv example')
args = parser.parse_args()
transformer_name = args.backend

factory = ngt.make_transformer_factory(transformer_name)
ngt.set_transformer_factory(factory)

print("\n--------- conv example using ", transformer_name, "-----------\n")

def test_conv():
    # This configuration has been tweaked to minimize overflow with 8.8 fixed point.
    N = 64  # 64 is the smallest possible for large N
    C, K = 4, 8  # C and K exchanged for f/bprop,  4 is the minimum that divides vec_size
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

    # randomly initialize
    input_value = rng.uniform(-.1, .1, ax_i)
    filter_value = rng.uniform(-.1, .1, ax_f)

    assert input_value.shape == ax_i.lengths
    assert filter_value.shape == ax_f.lengths

    inputs = ng.placeholder(ax_i)
    filters = ng.placeholder(ax_f)

    output = ng.convolution(conv_params, inputs, filters, axes=ax_o)
    targets = ng.placeholder(axes=output.axes)

    costs = ng.cross_entropy_binary(ng.sigmoid(output), targets)
    error = ng.sum(costs, reduction_axes=costs.axes) / ng.batch_size(costs)
    d_inputs = ng.deriv(error, inputs)
    d_filters = ng.deriv(error, filters)

    targets_value = rng.uniform(.1, 0.9, output.axes)

    trafo = ngt.make_transformer()
    conv_executor = trafo.computation([output, error, d_inputs, d_filters], inputs, filters, targets)
    result_ng, err_ng, gradI_ng, gradF_ng = conv_executor(input_value, filter_value, targets_value)

    print ("conv result", result_ng[:,0,3,3,63])
    print ("conv error", err_ng)
    print ("conv gradI", gradI_ng[:,0,3,3,63])
    print ("conv gradF", gradF_ng[2,0,:,:,7])

    return trafo


transformer = test_conv()

# if transformer_name == 'flexgpu' and transformer.flex_manager.num_flex_tensors < 20:
#     print(transformer.flex_manager.stat_ids)
#     fm = transformer.flex_manager

#     print(fm.host_stats)
#     fm.transfer_stats()
#     print(fm.host_stats)
