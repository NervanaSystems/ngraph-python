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
import os
import ngraph as ng
from benchmark import Benchmark
from examples.deepspeech.deepspeech import Deepspeech
from ngraph.frontends.neon import ax, GradientDescentMomentum, NgraphArgparser
from fake_data_generator import generate_ds2_data


def get_mini_ds2(inputs, nfilters, filter_width, str_w, nbands,
                 depth, hidden_size, batch_norm, device_id):
    model = Deepspeech(nfilters, filter_width, str_w, nbands, depth,
                       hidden_size, batch_norm=batch_norm, to_ctc=True)
    with ng.metadata(device_id=device_id, parallel=ax.N):
        model_out = model(inputs["audio"])
    return model_out


def run_mini_ds2_benchmark(max_length, nbands, str_w, nout, batch_size, max_iter, skip_iter,
                           nfilters, filter_width, depth, hidden_size, batch_norm, device_id,
                           device, transformer, nesterov=False, bprop=False, visualize=False):
    inputs, train_set, eval_set = generate_ds2_data(max_length, str_w, nout, nbands,
                                                    batch_size, max_iter)

    model_out = get_mini_ds2(inputs, nfilters, filter_width, str_w, nbands, depth, hidden_size,
                             batch_norm, device_id)

    if bprop:
        with ng.metadata(device_id=device_id, parallel=ax.N):
            loss = ng.ctc(model_out,
                          ng.flatten(inputs["char_map"]),
                          inputs["audio_length"],
                          inputs["trans_length"])

            optimizer = GradientDescentMomentum(learning_rate=2e-5,
                                                momentum_coef=0.99,
                                                gradient_clip_norm=400,
                                                nesterov=nesterov)

            updates = optimizer(loss)
            mean_cost = ng.sequential([updates, ng.mean(loss, out_axes=())])

            bprop_computation_op = ng.computation([mean_cost, model_out], "all")

        benchmark = Benchmark(bprop_computation_op, train_set, inputs, transformer, device)
        Benchmark.print_benchmark_results(benchmark.time(max_iter, skip_iter, 'ds2_bprop',
                                                         visualize, preprocess=True))
    else:
        fprop_computation_op = ng.computation(model_out, "all")

        benchmark_fprop = Benchmark(fprop_computation_op, train_set, inputs, transformer, device)
        Benchmark.print_benchmark_results(benchmark_fprop.time(max_iter, skip_iter, 'ds2_fprop',
                                                               visualize, preprocess=True))


if __name__ == "__main__":
    parser = NgraphArgparser(description='Train mini deep speech 2')
    parser.add_argument('--nfilters', type=int,
                        help='Number of convolutional filters in the first layer',
                        default=2)
    parser.add_argument('--filter_width', type=int,
                        help='Width of 1D convolutional filters',
                        default=11)
    parser.add_argument('--str_w', type=int,
                        help='Stride in time',
                        default=1)
    parser.add_argument('--depth', type=int,
                        help='Number of RNN layers',
                        default=1)
    parser.add_argument('--hidden_size', type=int,
                        help='Number of hidden units in the RNN and affine layers',
                        default=1)
    parser.add_argument('--nbands', type=int, default=13)
    parser.add_argument('--nout', type=int, default=29)
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--max_iter', type=int,
                        help='Number of  iterations',
                        default=2)
    parser.add_argument('-s', '--skip_iter', type=int,
                        help='Number of iterations to skip',
                        default=1)
    parser.add_argument('-n', '--num_devices', nargs='+', type=int,
                        help="number of devices to run the benchmark on",
                        default=[1])
    parser.add_argument('-d', '--hetr_device', choices=['cpu', 'gpu'],
                        help="device to run HeTr",
                        default='cpu')
    parser.add_argument('--max_length', type=float,
                        help="max duration for each audio sample",
                        default=0.3)
    parser.add_argument('--bprop', action="store_true", help="enable back propagation")
    parser.add_argument('--visualize', action="store_true", help="enable graph visualization")

    args = parser.parse_args()

    device_ids = [[str(device) for device in range(num_devices)]
                  for num_devices in args.num_devices]
    if args.hetr_device == 'gpu':
        os.environ["HETR_SERVER_GPU_NUM"] = str(len(device_ids[0]))

    ax.Y.length = args.nout
    ax.Y.name = "characters"

    for device_id in device_ids:
        run_mini_ds2_benchmark(max_length=args.max_length,
                               nbands=args.nbands,
                               str_w=args.str_w,
                               nout=args.nout,
                               batch_size=args.batch_size,
                               max_iter=args.max_iter,
                               skip_iter=args.skip_iter,
                               nfilters=args.nfilters,
                               filter_width=args.filter_width,
                               depth=args.depth,
                               hidden_size=args.hidden_size,
                               batch_norm=args.batch_norm,
                               device_id=device_id,
                               device=args.hetr_device,
                               transformer=args.backend,
                               nesterov=args.nesterov,
                               bprop=args.bprop,
                               visualize=args.visualize)
