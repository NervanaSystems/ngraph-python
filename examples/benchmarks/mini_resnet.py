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
"""
Run it using

python examples/benchmarks/mini_resnet.py -data cifar10 -b hetr -d cpu -m 2 -z 64 -t 2 -s 1 --bprop
"""
from __future__ import division
from __future__ import print_function
from .benchmark import Benchmark
from .fake_data_generator import generate_data
from ngraph.frontends.neon import GradientDescentMomentum
from ngraph.frontends.neon import ax, NgraphArgparser
from ngraph.frontends.neon import ArrayIterator
import ngraph as ng
from examples.resnet.resnet import BuildResnet


def get_mini_resnet(inputs, dataset, device, device_id, stage_depth=1,
                    batch_norm=False, activation=True, preprocess=False):
    en_bottleneck = False
    num_resnet_mods = 0
    if dataset == 'i1k':
        ax.Y.length = 1000
        if stage_depth > 34:
            en_bottleneck = True
    if dataset == 'cifar10':
        ax.Y.length = 10
        num_resnet_mods = (stage_depth - 2) // 6
    model = BuildResnet(dataset, stage_depth, en_bottleneck, num_resnet_mods,
                        batch_norm=batch_norm)
    with ng.metadata(device=device, device_id=device_id, parallel=ax.N):
        model_out = model(inputs['image'])
    return model_out


def get_fake_data(dataset, batch_size, num_iterations, seed=None):
    x_train, y_train = generate_data(dataset, batch_size, rand_seed=seed)

    train_data = {'image': {'data': x_train, 'axes': ('batch', 'C', 'H', 'W')},
                  'label': {'data': y_train, 'axes': ('batch',)}}

    train_set = ArrayIterator(train_data, batch_size, total_iterations=num_iterations)
    inputs = train_set.make_placeholders(include_iteration=True)
    return inputs, train_data, train_set


def run_resnet_benchmark(dataset, num_iterations, n_skip, batch_size, device_id,
                         transformer_type, device, bprop=True, batch_norm=False,
                         visualize=False, stage_depth=1):
    inputs, data, train_set = get_fake_data(dataset, batch_size, num_iterations)

    # Running forward propagation
    model_out = get_mini_resnet(inputs, dataset, device, device_id, batch_norm=batch_norm,
                                stage_depth=stage_depth)

    # Running back propagation
    if bprop:
        with ng.metadata(device=device, device_id=device_id, parallel=ax.N):
            optimizer = GradientDescentMomentum(0.01, 0.9)
            train_loss = ng.cross_entropy_multi(model_out,
                                                ng.one_hot(inputs['label'], axis=ax.Y))

            batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
            batch_cost_computation_op = ng.computation(batch_cost, "all")
        benchmark = Benchmark(batch_cost_computation_op, train_set, inputs,
                              transformer_type, device)
        Benchmark.print_benchmark_results(benchmark.time(num_iterations, n_skip,
                                                         dataset + '_msra_bprop',
                                                         visualize, 'device_id'))
    else:
        fprop_computation_op = ng.computation(model_out, 'all')
        benchmark = Benchmark(fprop_computation_op, train_set, inputs,
                              transformer_type, device)
        Benchmark.print_benchmark_results(benchmark.time(num_iterations, n_skip,
                                                         dataset + '_msra_fprop',
                                                         visualize))


if __name__ == "__main__":
    parser = NgraphArgparser(description='Train deep residual network')
    parser.add_argument('-data', '--data_set', default='cifar10',
                        choices=['cifar10', 'i1k'], help="data set name")
    parser.add_argument('-s', '--skip_iter', type=int, default=1,
                        help="number of iterations to skip")
    parser.add_argument('-m', '--num_devices', nargs='+', type=int, default=[1],
                        help="number of devices to run the benchmark on")
    parser.add_argument('--hetr_device', default='cpu', choices=['cpu', 'gpu'],
                        help="device to run HeTr")
    parser.add_argument('--bprop', action="store_true", help="enable back propagation")
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='whether to use batch normalization')
    parser.add_argument('--graph_vis', action="store_true", help="enable graph visualization")
    parser.add_argument('--size', type=int, default=18, help="Enter size of resnet")
    args = parser.parse_args()

    device_ids = [[str(device) for device in range(num_devices)]
                  for num_devices in args.num_devices]
    for device_id in device_ids:
        run_resnet_benchmark(dataset=args.data_set,
                             num_iterations=args.num_iterations,
                             n_skip=args.skip_iter,
                             batch_size=args.batch_size,
                             device_id=device_id,
                             transformer_type=args.backend,
                             device=args.hetr_device,
                             bprop=args.bprop,
                             batch_norm=args.use_batch_norm,
                             visualize=args.graph_vis,
                             stage_depth=args.size)
