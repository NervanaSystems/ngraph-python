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
Convnet-Alexnet Benchmark with spelled out neon model framework in one file
https://github.com/soumith/convnet-benchmarks

./alexnet.py

"""

import numpy as np
import ngraph as ng
import ngraph.transformers as ngt
from contextlib import closing

from ngraph.frontends.neon import NgraphArgparser, ArrayIterator, GaussianInit
from ngraph.frontends.neon import Affine, Convolution, Pool2D, Sequential
from ngraph.frontends.neon import Rectlin, Softmax, GradientDescentMomentum
from ngraph.frontends.neon import make_bound_computation, make_default_callbacks
from ngraph.frontends.neon import loop_train, ax

np.seterr(all='raise')

parser = NgraphArgparser(description='Train convnet-alexnet model on random dataset')
# Default batch_size for convnet-alexnet is 128.
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

# Setup model
seq1 = Sequential([Convolution((11, 11, 64), filter_init=GaussianInit(var=0.01),
                               activation=Rectlin(), padding=3, strides=4),
                   Pool2D(3, strides=2),
                   Convolution((5, 5, 192), filter_init=GaussianInit(var=0.01),
                               activation=Rectlin(), padding=2),
                   Pool2D(3, strides=2),
                   Convolution((3, 3, 384), filter_init=GaussianInit(var=0.03),
                               activation=Rectlin(), padding=1),
                   Convolution((3, 3, 256), filter_init=GaussianInit(var=0.03),
                               activation=Rectlin(), padding=1),
                   Convolution((3, 3, 256), filter_init=GaussianInit(var=0.03),
                               activation=Rectlin(), padding=1),
                   Pool2D(3, strides=2),
                   Affine(nout=4096, weight_init=GaussianInit(var=0.01),
                          activation=Rectlin()),
                   Affine(nout=4096, weight_init=GaussianInit(var=0.01),
                          activation=Rectlin()),
                   Affine(nout=1000, weight_init=GaussianInit(var=0.01),
                          activation=Softmax())])

# Learning rate change based on schedule from learning_rate_policies.py
lr_schedule = {'name': 'schedule', 'base_lr': 0.01,
               'gamma': (1 / 250.)**(1 / 3.),
               'schedule': [22, 44, 65]}
optimizer = GradientDescentMomentum(lr_schedule, 0.0, wdecay=0.0005,
                                    iteration=inputs['iteration'])
train_prob = seq1(inputs['image'])
train_loss = ng.cross_entropy_multi(train_prob, ng.one_hot(inputs['label'], axis=ax.Y))
batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_outputs = dict(batch_cost=batch_cost)

with closing(ngt.make_transformer()) as transformer:
    train_computation = make_bound_computation(transformer, train_outputs, inputs)

    cbs = make_default_callbacks(output_file=args.output_file,
                                 frequency=10,
                                 train_computation=train_computation,
                                 total_iterations=args.num_iterations)

    loop_train(train_set, train_computation, cbs)
