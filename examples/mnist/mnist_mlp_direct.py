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
from ngraph.frontends.neon import nnAffine, Rectlin, Logistic, SimpleModel, ModelTrainer, GDMopt
import argparse
from functools import partial
from data import make_aeon_loaders


class GaussianInit(object):
    def __init__(self, mean=0.0, var=0.01):
        self.functor = partial(np.random.normal, mean, var)

    def __call__(self, out_shape):
        return self.functor(out_shape)


class UniformInit(object):
    def __init__(self, low=-0.01, high=0.01):
        self.functor = partial(np.random.uniform, low, high)

    def __call__(self, out_shape):
        return self.functor(out_shape)


class ConstantInit(object):
    def __init__(self, val=0.0):
        self.val = val

    def __call__(self, out_shape):
        return val


parser = argparse.ArgumentParser(description='Ingest MNIST from pkl to pngs')
parser.add_argument('--train')
parser.add_argument('--valid')
parser.add_argument('--results_file', default='results.csv')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_iterations', type=int, default=400)
parser.add_argument('--iter_interval', type=int, default=200)
parser.add_argument('--rseed', default=0)
args = parser.parse_args()

np.random.seed(args.rseed)

######################
# Model specification
hidden_size, output_size = 100, 10
H1 = ng.Axis(hidden_size, name="H1")
Y = ng.Axis(output_size, name="Y")

my_model = SimpleModel([nnAffine(out_axis=H1, init=GaussianInit(), activation=Rectlin()),
                        nnAffine(out_axis=Y, init=GaussianInit(), activation=Logistic())])


# Create the dataloader
train_set, valid_set = make_aeon_loaders(args.train, args.valid, args.batch_size)

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


def classify_cost(pred, targ):
    return ng.cross_entropy_binary(pred, ng.Onehot(targ, axis=pred.axes.sample_axes()[0]))


def misclass_error(pred, targ):
    return ng.not_equal(ng.argmax(pred), targ)


trainer = ModelTrainer(args.num_iterations, args.iter_interval)

trainer.make_train_graph(my_model, x, t, classify_cost, GDMopt(0.1, 0.9))
trainer.make_inference_graph(my_model, x)

trainer.bind_transformer(ng.NumPyTransformer())

# train
trainer.train(train_set)

# # validate
hyps, refs = trainer.eval(valid_set)
np.savetxt(args.results_file, [(h,r) for h, r in zip(hyps, refs)], fmt='%s,%s')
