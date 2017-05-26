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
Convnet-overfeat Benchmark with spelled out neon model framework in one file
https://github.com/soumith/convnet-benchmarks

./overfeat.py

"""

import numpy as np
import ngraph as ng
import ngraph.transformers as ngt
from tqdm import tqdm

from ngraph.frontends.neon import NgraphArgparser, ArrayIterator, GaussianInit
from ngraph.frontends.neon import Affine, Convolution, Pool2D, Sequential
from ngraph.frontends.neon import Rectlin, Softmax, GradientDescentMomentum
from ngraph.frontends.neon import ax

np.seterr(all='raise')

parser = NgraphArgparser(description='Train convnet-overfeat model on random dataset')
# Default batch_size for convnet-overfeat is 128.
parser.set_defaults(batch_size=128, num_iterations=100)
args = parser.parse_args()

# Setup data provider
image_size = 231
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
seq1 = Sequential([Convolution((11, 11, 96), filter_init=GaussianInit(var=0.01),
                               activation=Rectlin(), padding=0, strides=4),
                   Pool2D(2, strides=2),
                   Convolution((5, 5, 256), filter_init=GaussianInit(var=0.01),
                               activation=Rectlin(), padding=0),
                   Pool2D(2, strides=2),
                   Convolution((3, 3, 512), filter_init=GaussianInit(var=0.01),
                               activation=Rectlin(), padding=1),
                   Convolution((3, 3, 1024), filter_init=GaussianInit(var=0.01),
                               activation=Rectlin(), padding=1),
                   Convolution((3, 3, 1024), filter_init=GaussianInit(var=0.01),
                               activation=Rectlin(), padding=1),
                   Pool2D(2, strides=2),
                   Affine(nout=3072, weight_init=GaussianInit(var=0.01),
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
train_computation = ng.computation(batch_cost, "all")

# Now bing the computations we are interested in
transformer = ngt.make_transformer()
train_function = transformer.add_computation(train_computation)

tpbar = tqdm(unit="batches", ncols=100, total=args.num_iterations)
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
