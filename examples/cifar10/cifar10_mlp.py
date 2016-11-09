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
MNIST MLP with spelled out neon model framework in one file

The motivation is to show the flexibility of ngraph and how user can build a
model without the neon architecture. This may also help with debugging.

Run it using

python examples/mnist/mnist_mlp.py --data_dir /usr/local/data/MNIST --output_file out.hd5

"""
from __future__ import division
from __future__ import print_function
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import nnAffine, nnPreprocess, Sequential, Callbacks
from ngraph.frontends.neon import UniformInit, Rectlin, Softmax, GradientDescentMomentum
from ngraph.frontends.neon import ax, make_keyed_computation
from ngraph.frontends.neon import NgraphArgparser
from ngraph.frontends.neon import ArrayIterator

from cifar10 import CIFAR10
import ngraph.transformers as ngt

parser = NgraphArgparser(description='Train simple mlp on cifar10 dataset')
args = parser.parse_args()

np.random.seed(args.rng_seed)

# Create the dataloader
train_data, valid_data = CIFAR10(args.data_dir).load_data()
train_set = ArrayIterator(train_data, args.batch_size, total_iterations=args.num_iterations)
valid_set = ArrayIterator(valid_data, args.batch_size)

######################
# Model specification


def cifar_mean_subtract(x):
    bgr_mean = ng.persistent_tensor(axes=x.axes[0], initial_value=np.array([[104, 119, 127]]))
    return (x - bgr_mean) / 255.

seq1 = Sequential([nnPreprocess(functor=cifar_mean_subtract),
                   nnAffine(nout=200, init=UniformInit(-0.1, 0.1), activation=Rectlin()),
                   nnAffine(axes=ax.Y, init=UniformInit(-0.1, 0.1), activation=Softmax())])

######################
# Input specification
ax.C.length, ax.H.length, ax.W.length = train_set.shapes[0]
ax.N.length = args.batch_size
ax.Y.length = 10


# placeholders with descriptive names
inputs = dict(img=ng.placeholder([ax.C, ax.H, ax.W, ax.N]),
              tgt=ng.placeholder([ax.N]),
              idx=ng.placeholder([]))

optimizer = GradientDescentMomentum(0.1, 0.9)
output_prob = seq1.train_outputs(inputs['img'])
errors = ng.not_equal(ng.argmax(output_prob, out_axes=[ax.N]), inputs['tgt'])
train_cost = ng.cross_entropy_multi(output_prob, ng.one_hot(inputs['tgt'], axis=ax.Y))
mean_cost = ng.mean(train_cost, out_axes=())
updates = optimizer(train_cost, inputs['idx'])


# Now bind the computations we are interested in
transformer = ngt.make_transformer()
train_computation = make_keyed_computation(transformer, [mean_cost, updates], inputs)
inference_computation = make_keyed_computation(transformer, errors, inputs)

cb = Callbacks(seq1, args.output_file, args.iter_interval)

######################
# Train Loop
cb.on_train_begin(args.num_iterations)

for mb_idx, data in enumerate(train_set):
    cb.on_minibatch_begin(mb_idx)
    batch_cost, _ = train_computation(dict(img=data[0], tgt=data[1], idx=mb_idx))
    seq1.current_batch_cost = float(batch_cost)
    cb.on_minibatch_end(mb_idx)

cb.on_train_end()

######################
# Evaluation
all_errors = []
for data in valid_set:
    batch_errors = inference_computation(dict(img=data[0], tgt=data[1], idx=0))
    bsz = min(valid_set.ndata - len(batch_errors), len(batch_errors))
    all_errors.extend(list(batch_errors[:bsz]))

print("Misclassification: {}".format(np.mean(all_errors)))
