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
from __future__ import print_function
from ngraph.frontends.neon import ax, np, Affine, Axes, Callbacks, CrossEntropyMulti,\
    GeneralizedCost, GradientDescentMomentum, Misclassification, Model,\
    NgraphArgparser, Rectlin, Softmax

from neon.initializers import Uniform
from neon.data import CIFAR10

# parse the command line arguments (generates the backend)
parser = NgraphArgparser(__doc__)
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()

# setup data provider
dataset = CIFAR10(path=args.data_dir,
                  normalize=True,
                  contrast_normalize=False,
                  whiten=False)
train = dataset.train_iter
test = dataset.valid_iter

init_uni = Uniform(low=-0.1, high=0.1)
opt_gdm = GradientDescentMomentum(learning_rate=0.01, momentum_coef=0.9)


# set up the model layers
layers = [
    Affine(nout=200, init=init_uni, activation=Rectlin()),
    Affine(nout=10, axes=Axes(ax.Y,), init=init_uni,
           activation=Softmax()),
]

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

mlp = Model(layers=layers)
callbacks = Callbacks(mlp, eval_set=test, **args.callback_args)
mlp.initialize(
    dataset=train,
    input_axes=Axes((ax.C, ax.H, ax.W)),
    target_axes=Axes((ax.Y,)),
    optimizer=opt_gdm,
    cost=cost,
    metric=Misclassification()
)

np.seterr(divide='raise', over='raise', invalid='raise')
mlp.fit(
    train,
    num_epochs=args.epochs,
    callbacks=callbacks
)

print('Misclassification error = %.1f%%' %
      (mlp.eval(test) * 100))
