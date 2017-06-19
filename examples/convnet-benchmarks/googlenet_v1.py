#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
Convnet-GoogLeNet v1 Benchmark with spelled out neon model framework in one file
https://github.com/soumith/convnet-benchmarks

./googlenet_v1.py

"""

import numpy as np
import ngraph as ng
import ngraph.transformers as ngt
from tqdm import tqdm
from contextlib import closing

from ngraph.frontends.neon import NgraphArgparser, ArrayIterator
from ngraph.frontends.neon import XavierInit, UniformInit
from ngraph.frontends.neon import Affine, Convolution, Pool2D, Sequential
from ngraph.frontends.neon import Rectlin, Softmax, GradientDescentMomentum
from ngraph.frontends.neon import ax

np.seterr(all='raise')

parser = NgraphArgparser(description=__doc__)
# Default batch_size for convnet-googlenet is 128.
parser.set_defaults(batch_size=128, num_iterations=100)
args = parser.parse_args()

# Setup data provider
image_size = 224
X_train = np.random.uniform(-1, 1, (args.batch_size, 3, image_size, image_size))
y_train = np.ones(shape=(args.batch_size), dtype=np.int32)
train_data = {'image': {'data': X_train,
                        'axes': ('batch', 'C', 'height', 'width')},
              'label': {'data': y_train,
                        'axes': ('batch',)}}
train_set = ArrayIterator(train_data,
                          batch_size=args.batch_size,
                          total_iterations=args.num_iterations)
inputs = train_set.make_placeholders(include_iteration=True)
ax.Y.length = 1000  # number of outputs of last layer.

# weight initialization
bias_init = UniformInit(low=-0.08, high=0.08)


class Inception(Sequential):

    def __init__(self, branch_units, activation=Rectlin(),
                 bias_init=UniformInit(low=-0.08, high=0.08),
                 filter_init=XavierInit()):

        (p1, p2, p3, p4) = branch_units

        self.branch_1 = Convolution((1, 1, p1[0]), activation=activation,
                                    bias_init=bias_init,
                                    filter_init=filter_init)
        self.branch_2 = [Convolution((1, 1, p2[0]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init),
                         Convolution((3, 3, p2[1]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding=1)]
        self.branch_3 = [Convolution((1, 1, p3[0]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init),
                         Convolution((5, 5, p3[1]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding=2)]
        self.branch_4 = [Pool2D(fshape=3, padding=1, strides=1, op="max"),
                         Convolution((1, 1, p3[0]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init)]

    def __call__(self, in_obj):

        branch_1_output = self.branch_1(in_obj)
        branch_2_output = self.branch_2[0](in_obj)
        branch_2_output = self.branch_2[1](branch_2_output)
        branch_3_output = self.branch_3[0](in_obj)
        branch_3_output = self.branch_3[1](branch_3_output)
        branch_4_output = self.branch_4[0](in_obj)
        branch_4_output = self.branch_4[1](branch_4_output)

        outputs = [branch_1_output, branch_2_output, branch_3_output, branch_4_output]
        # This does the equivalent of neon's merge-broadcast
        return ng.concat_along_axis(outputs, branch_1_output.axes.channel_axis())


seq1 = Sequential([Convolution((7, 7, 64), padding=3, strides=2,
                               activation=Rectlin(), bias_init=bias_init,
                               filter_init=XavierInit()),
                   Pool2D(fshape=3, padding=1, strides=2, op='max'),
                   Convolution((1, 1, 64), activation=Rectlin(),
                               bias_init=bias_init, filter_init=XavierInit()),
                   Convolution((3, 3, 192), activation=Rectlin(),
                               bias_init=bias_init, filter_init=XavierInit(),
                               padding=1),
                   Pool2D(fshape=3, padding=1, strides=2, op='max'),
                   Inception([(64,), (96, 128), (16, 32), (32,)]),
                   Inception([(128,), (128, 192), (32, 96), (64,)]),
                   Pool2D(fshape=3, padding=1, strides=2, op='max'),
                   Inception([(192,), (96, 208), (16, 48), (64,)]),
                   Inception([(160,), (112, 224), (24, 64), (64,)]),
                   Inception([(128,), (128, 256), (24, 64), (64,)]),
                   Inception([(112,), (144, 288), (32, 64), (64,)]),
                   Inception([(256,), (160, 320), (32, 128), (128,)]),
                   Pool2D(fshape=3, padding=1, strides=2, op='max'),
                   Inception([(256,), (160, 320), (32, 128), (128,)]),
                   Inception([(384,), (192, 384), (48, 128), (128,)]),
                   Pool2D(fshape=7, strides=1, op="avg"),
                   Affine(axes=ax.Y, weight_init=XavierInit(),
                          bias_init=bias_init, activation=Softmax())])

lr_schedule = {'name': 'schedule', 'base_lr': 0.01,
               'gamma': (1 / 250.)**(1 / 3.),
               'schedule': [22, 44, 65]}

optimizer = GradientDescentMomentum(lr_schedule, 0.0, wdecay=0.0005,
                                    iteration=inputs['iteration'])
train_prob = seq1(inputs['image'])
train_loss = ng.cross_entropy_multi(train_prob, ng.one_hot(inputs['label'], axis=ax.Y))
batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_computation = ng.computation(batch_cost, 'all')

with closing(ngt.make_transformer()) as transformer:
    train_function = transformer.add_computation(train_computation)

    if args.no_progress_bar:
        ncols = 0
    else:
        ncols = 100

    tpbar = tqdm(unit="batches", ncols=ncols, total=args.num_iterations)
    interval_cost = 0.0

    for step, data in enumerate(train_set):
        data['iteration'] = step
        feed_dict = {inputs[k]: data[k] for k in inputs.keys()}
        output = train_function(feed_dict=feed_dict)

        tpbar.update(1)
        tpbar.set_description("Training {:0.4f}".format(output[()]))
        interval_cost += output[()]
        if (step + 1) % args.iter_interval == 0 and step > 0:
            tqdm.write("Interval {interval} Iteration {iteration} complete. "
                       "Avg Train Cost {cost:0.4f}".format(
                           interval=step // args.iter_interval,
                           iteration=step,
                           cost=interval_cost / args.iter_interval))
