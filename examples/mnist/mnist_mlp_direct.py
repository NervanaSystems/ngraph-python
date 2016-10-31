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
python examples/mnist/mnist_mlp_direct.py --train /scratch/alex/MNIST/manifest_train.csv \
      --valid /scratch/alex/MNIST/manifest_valid.csv

      for examples
"""
from __future__ import division
from __future__ import print_function
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import nnAffine, nnPreprocess, Model, Callbacks
from ngraph.frontends.neon import GaussianInit, Rectlin, Logistic, GradientDescentMomentum
import argparse
from data import make_aeon_loaders


parser = argparse.ArgumentParser(description='Ingest MNIST from pkl to pngs')
parser.add_argument('--work_dir', required=True)
parser.add_argument('--output_file')
parser.add_argument('--results_file', default='results.csv')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_iterations', type=int, default=2000)
parser.add_argument('--iter_interval', type=int, default=200)
parser.add_argument('--rseed', default=0)
args = parser.parse_args()

np.random.seed(args.rseed)

######################
# Model specification
hidden_size, output_size = 100, 10
H1 = ng.Axis(hidden_size, name="H1")
H2 = ng.Axis(hidden_size, name="H2")
Y = ng.Axis(output_size, name="Y")

def unit_scale_mnist_pixels(x):
    return x / 255.

my_model = Model([nnPreprocess(functor=unit_scale_mnist_pixels),
                  nnAffine(out_axis=H1, init=GaussianInit(), activation=Rectlin()),
                  nnAffine(out_axis=H2, init=GaussianInit(), activation=Rectlin()),
                  nnAffine(out_axis=Y, init=GaussianInit(), activation=Logistic())])


transformer = ng.NumPyTransformer()

# Create the dataloader
train_set, valid_set = make_aeon_loaders(args.work_dir, args.batch_size, transformer)

######################
# Input specification
image_channels, image_height, image_width = train_set.shapes()[0]
C = ng.Axis(image_channels, name="C")
H = ng.Axis(image_height, name="H")
W = ng.Axis(image_width, name="W")
N = ng.Axis(args.batch_size, name="N", batch=True)

# place holder
x = ng.placeholder(axes=ng.Axes([C, H, W, N]))
t = ng.placeholder(axes=ng.Axes([N]))
it_idx = ng.placeholder(axes=ng.Axes())  # iteration index, for learning rate evolution

optimizer = GradientDescentMomentum(0.1, 0.9)

pred = my_model.get_outputs(x)

train_cost = ng.cross_entropy_binary(pred, ng.Onehot(t, axis=Y))
train_graph = ([ng.mean(train_cost, out_axes=()),  # mean cost for display
                optimizer(train_cost, it_idx)],  # update function for optimization
               x, t, it_idx)  # inputs

inf_graph = (pred, x)


my_model.bind_transformer(transformer, train_graph, inf_graph)

# train
cb = Callbacks(my_model, args.output_file, args.iter_interval)
my_model.train(train_set, args.num_iterations, cb)

# # validate
hyps, refs = my_model.eval(valid_set)
np.savetxt(args.results_file, [(h,r) for h, r in zip(hyps, refs)], fmt='%s,%s')
a = np.loadtxt(args.results_file, delimiter=',')
err = np.sum((a[:,0]!= a[:,1]))/float(a.shape[0])
print("Misclassification: {}".format(err))
