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

python examples/mnist/mnist_mlp.py --work_dir /usr/local/data/MNIST --output_file out.hd5

"""
from __future__ import division
from __future__ import print_function
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import nnAffine, nnPreprocess, Sequential, Callbacks
from ngraph.frontends.neon import GaussianInit, Rectlin, Logistic, GradientDescentMomentum
from ngraph.frontends.neon import ax, make_keyed_computation
import argparse
# from data import make_aeon_loaders
from aeon import SimpleDataLoader
from ngraph.frontends.neon import ArrayIterator

from mnist import MNIST
from ngraph.util.utils import executor
from ngraph.transformers import Transformer

parser = argparse.ArgumentParser(description='Train simple mlp on mnist dataset')
parser.add_argument('--work_dir', required=True)
parser.add_argument('--output_file')
parser.add_argument('--results_file', default='results.csv')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_iterations', type=int, default=2000)
parser.add_argument('--iter_interval', type=int, default=200)
parser.add_argument('--rseed', type=int, default=0)
args = parser.parse_args()

np.random.seed(args.rseed)

# Create the dataloader
train_data, valid_data = MNIST(args.work_dir).load_data()
# train_set = SimpleDataLoader(train_data, args.batch_size, total_iterations=args.num_iterations)
train_set = ArrayIterator(train_data, args.batch_size, total_iterations=args.num_iterations)
# train_set, valid_set = make_aeon_loaders(args.work_dir, args.batch_size, transformer)


######################
# Model specification
seq1 = Sequential([nnPreprocess(functor=lambda x: x / 255.),
                   nnAffine(nout=100, init=GaussianInit(), activation=Rectlin()),
                   nnAffine(axes=ax.Y, init=GaussianInit(), activation=Logistic())])

######################
# Input specification
ax.C.length, ax.H.length, ax.W.length = train_set.shapes[0]
ax.N.length = args.batch_size
ax.Y.length = 10


# placeholders
inputs = dict(img=ng.placeholder(axes=ng.make_axes([ax.C, ax.H, ax.W, ax.N])),
              tgt=ng.placeholder(axes=ng.make_axes([ax.N])),
              idx=ng.placeholder(axes=ng.make_axes()))

optimizer = GradientDescentMomentum(0.1, 0.9)
pred = seq1.train_outputs(inputs['img'])
train_cost = ng.cross_entropy_binary(pred, ng.Onehot(inputs['tgt'], axis=ax.Y))
mean_cost = ng.mean(train_cost, out_axes=())
updates = optimizer(train_cost, inputs['idx'])

Transformer.make_transformer()

train_computation = make_keyed_computation(executor, [mean_cost, updates], inputs)

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


# # validate
# hyps, refs = my_model.eval(valid_set)
# np.savetxt(args.results_file, [(h, r) for h, r in zip(hyps, refs)], fmt='%s,%s')
# a = np.loadtxt(args.results_file, delimiter=',')
# err = np.sum((a[:, 0] != a[:, 1])) / float(a.shape[0])
# print("Misclassification: {}".format(err))
