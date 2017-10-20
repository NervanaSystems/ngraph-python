#!/usr/bin/env python
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
"""
MNIST MLP with spelled out neon model framework in one file

The motivation is to show the flexibility of ngraph and how user can build a
model without the neon architecture. This may also help with debugging.

Run it using

python examples/mnist/mnist_mlp.py --data_dir /usr/local/data/MNIST --output_file out.hd5

"""
from __future__ import division
from __future__ import print_function
from contextlib import closing
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Layer, Affine, Preprocess, Sequential, Softmax, Convolution
from ngraph.frontends.neon import GaussianInit, Rectlin, Logistic, GradientDescentMomentum
from ngraph.frontends.neon import ax, loop_train, make_bound_computation, make_default_callbacks
from ngraph.frontends.neon import NgraphArgparser, Pooling
from ngraph.frontends.neon import ArrayIterator

from ngraph.frontends.neon import MNIST
import ngraph.transformers as ngt

parser = NgraphArgparser(description='Train simple mlp on mnist dataset')
args = parser.parse_args()

np.random.seed(args.rng_seed)

# Create the dataloader
train_data, valid_data = MNIST(args.data_dir).load_data()
train_set = ArrayIterator(train_data, args.batch_size, total_iterations=args.num_iterations)
valid_set = ArrayIterator(valid_data, args.batch_size)

inputs = train_set.make_placeholders()
ax.Y.length = 10

np.random.seed(1)
######################
# Model specification
seq1 = Sequential([Preprocess(functor=lambda x: x / 255.),
                   Convolution(filter_shape=(3, 3, 15), padding = 1, batch_norm=True, filter_init=GaussianInit(), activation=Rectlin()),
                   Pooling(pool_shape=(4, 4), padding=0, strides=4, pool_type='avg'),
                   Convolution(filter_shape=(3, 3, 15), padding = 1, batch_norm=False, filter_init=GaussianInit(), activation=Rectlin()),
                   Pooling(pool_shape=(7, 7), padding=0, pool_type='avg'),
                   Convolution(filter_shape=(1, 1, 10), padding = 0, filter_init=GaussianInit(), activation=Softmax())])

optimizer = GradientDescentMomentum(0.1, 0.9)
train_prob = seq1(inputs['image'])
train_prob= train_prob[:,:,0,0]
train_prob = ng.map_roles(train_prob, {"C": ax.Y.name})
train_loss = ng.cross_entropy_multi(train_prob, ng.one_hot(inputs['label'], axis=ax.Y))

batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_outputs = dict(batch_cost=batch_cost)

with Layer.inference_mode_on():
    inference_prob = seq1(inputs['image'])[:,:,0,0]
inference_prob = ng.map_roles(inference_prob, {"C": ax.Y.name})
errors = ng.not_equal(ng.argmax(inference_prob, out_axes=[ax.N]), inputs['label'])
eval_loss = ng.cross_entropy_multi(inference_prob, ng.one_hot(inputs['label'], axis=ax.Y))
eval_outputs = dict(cross_ent_loss=eval_loss, misclass_pct=errors)

# Now bind the computations we are interested in
with closing(ngt.make_transformer()) as transformer:
    train_computation = make_bound_computation(transformer, train_outputs, inputs)
    loss_computation = make_bound_computation(transformer, eval_outputs, inputs)
    
    interval_cost = 0.
    for iter_no, data in enumerate(train_set):
        iter_cost = train_computation(data)
        interval_cost += iter_cost['batch_cost']
        if (iter_no+1) % args.iter_interval == 0:
            interval_cost = interval_cost / args.iter_interval
            print("Iteration:%d, Interval Cost: %.5e" % (iter_no, interval_cost))
            interval_cost = 0.
