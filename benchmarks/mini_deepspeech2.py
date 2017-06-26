#!/usr/bin/env python
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
import ngraph as ng
from benchmark import Benchmark
from ngraph.frontends.neon import (GaussianInit, GlorotInit, ConstantInit, Convolution, Rectlin, Rectlinclip,
                                   BiRNN, Affine, Softmax, Sequential)

from ngraph.frontends.neon import ax
from fake_data_generator import generate_ds2_data
import argparse

ax.Y.length = 29
ax.Y.name = "characters"

class DeepBiRNN(Sequential):

    def __init__(self, num_layers, hidden_size, init, activation, batch_norm=False,
                 sum_final=False):

        rnns = list()
        sum_out = concat_out = False
        for ii in range(num_layers):
            if ii == (num_layers - 1):
                sum_out = sum_final
                concat_out = not sum_final

            rnn = BiRNN(hidden_size, init=init,
                        activation=activation,
                        batch_norm=batch_norm,
                        reset_cells=True, return_sequence=True,
                        concat_out=concat_out, sum_out=sum_out)
            rnns.append(rnn)

        super(DeepBiRNN, self).__init__(layers=rnns)

    def __call__(self, in_obj, *args, **kwargs):

        # TODO: Is this how we want to do this?
        if in_obj.axes.recurrent_axis() is None:
            in_obj = ng.map_roles(in_obj, {"time": "REC"})
            assert in_obj.axes.recurrent_axis() is not None, "in_obj has no recurrent or time axis"

        return super(DeepBiRNN, self).__call__(in_obj, *args, **kwargs)

    @property
    def params(self):

        params = list()
        for rnn in self.layers:
            params.extend([rnn.fwd_rnn.W_recur,
                           rnn.bwd_rnn.W_recur,
                           rnn.fwd_rnn.W_input,
                           rnn.bwd_rnn.W_input])

        return params


class Deepspeech(Sequential):

    def __init__(self, nfilters, filter_width, str_w, nbands, depth, hidden_size,
                 batch_norm=False, batch_norm_affine=False, batch_norm_conv=False, to_ctc=False):

        self.to_ctc = to_ctc

        # Initializers
        gauss = GaussianInit(0.01)
        glorot = GlorotInit()

        # 1D Convolution layer
        padding = dict(pad_h=0, pad_w=filter_width // 2, pad_d=0)
        strides = dict(str_h=1, str_w=str_w, str_d=1)
        dilation = dict(dil_d=1, dil_h=1, dil_w=1)

        conv_layer = Convolution((nbands, filter_width, nfilters),
                                 gauss,
                                 bias_init=ConstantInit(0),
                                 padding=padding,
                                 strides=strides,
                                 dilation=dilation,
                                 activation=Rectlin(),
                                 batch_norm=batch_norm_conv)

        # Add BiRNN layers
        deep_birnn = DeepBiRNN(depth, hidden_size, glorot, Rectlinclip(), batch_norm=batch_norm)

        # Add a single affine layer
        fc = Affine(nout=hidden_size, weight_init=glorot,
                    activation=Rectlinclip(),
                    batch_norm=batch_norm_affine)

        # Add the final affine layer
        # Softmax output is computed within the CTC cost function, so no activation is needed here.
        if self.to_ctc is False:
            activation = Softmax()
        else:
            activation = None
        final = Affine(axes=ax.Y, weight_init=glorot, activation=activation)

        layers = [conv_layer,
                  deep_birnn,
                  fc,
                  final]

        super(Deepspeech, self).__init__(layers=layers)

    def __call__(self, *args, **kwargs):

        output = super(Deepspeech, self).__call__(*args, **kwargs)
        # prepare activations/gradients for warp-ctc
        # TODO: This should be handled in a graph pass
        if self.to_ctc is True:
            warp_axes = ng.make_axes([output.axes.recurrent_axis(),
                                      output.axes.batch_axis()]) | \
                        output.axes.feature_axes()
            output = ng.axes_with_order(output, warp_axes)
            output = ng.ContiguousOp(output)

        return output

    @property
    def params(self):

        params = [self.layers[0].conv.W]
        params.extend(self.layers[1].params)
        params.append(self.layers[2].linear.W)
        params.append(self.layers[3].linear.W)

        return params


def get_mini_ds2(inputs, nfilters, filter_width, str_w, nbands, depth, hidden_size, batch_norm, device_id):
    model = Deepspeech(nfilters, filter_width, str_w, nbands, depth, hidden_size, batch_norm=batch_norm)
    with ng.metadata(device_id=device_id, parallel=ax.N):
        model_out = model(inputs["audio"])
    return model_out


def run_mini_ds2_benchmark(max_length, nbands, nout, str_w, batch_size, max_iter, skip_iter, nfilters,
                           filter_width, depth, hidden_size, batch_norm, device_id, device, transformer, visualize=False):
    inputs, train_set, eval_set = generate_ds2_data(max_length, nbands, nout, str_w,
                                                    batch_size, max_iter)
    model_out = get_mini_ds2(inputs, nfilters, filter_width, str_w, nbands, depth, hidden_size,
                             batch_norm, device_id)

    fprop_computation_op = ng.computation(model_out, "all")

    benchmark_fprop = Benchmark(fprop_computation_op, train_set, inputs, transformer, device)
    Benchmark.print_benchmark_results(benchmark_fprop.time(max_iter, skip_iter,
                                                           'ds2_fprop', visualize))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nfilters', type=int,
                           help='Number of convolutional filters in the first layer',
                           default=2)
    parser.add_argument('--filter_width', type=int,
                           help='Width of 1D convolutional filters',
                           default=11)
    parser.add_argument('--str_w', type=int,
                           help='Stride in time',
                           default=3)
    parser.add_argument('--depth', type=int,
                           help='Number of RNN layers',
                           default=1)
    parser.add_argument('--hidden_size', type=int,
                           help='Number of hidden units in the RNN and affine layers',
                           default=1)
    parser.add_argument('--nbands', type=int, default=13)
    parser.add_argument('--nout', type=int, default=29)
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--max_iter', type=int, help='Number of  iterations', default=2)
    parser.add_argument('--skip_iter', type=int, help='Number of iterations to skip', default=1)
    parser.add_argument('-n', '--num_devices', nargs='+', type=int, default=[1],
                        help="number of devices to run the benchmark on")
    parser.add_argument('--batch_size', type=int, help='Batch size', default=64)
    parser.add_argument('--transformer', default='hetr', help='Type of Transformer')
    parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'gpu'],
                        help="device to run on")
    parser.add_argument('--max_length', type=float,
                             help="max duration for each audio sample",
                             default=0.03)
    parser.add_argument('-v', '--visualize', action="store_true",
                        help="enable graph visualization")

    args = parser.parse_args()

    device_ids = [[str(device) for device in range(num_devices)]
                  for num_devices in args.num_devices]

    device_id=('1','2')
    run_mini_ds2_benchmark(args.max_length, args.nbands, args.nout, args.str_w, args.batch_size, args.max_iter, args.skip_iter, args.nfilters, args.filter_width,
                               args.depth, args.hidden_size, args.batch_norm, device_id, args.device, args.transformer, args.visualize)