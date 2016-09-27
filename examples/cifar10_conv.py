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
from ngraph.frontends.neon import ax, np, Affine, Conv, Axes, Callbacks, CrossEntropyMulti,\
    GeneralizedCost, GradientDescentMomentum, Misclassification, Model,\
    NeonArgparser, Rectlin, Softmax, Uniform

from neon.data import CIFAR10

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.set_defaults(backend='dataloader')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
parser.set_defaults(backend='dataloader')
args = parser.parse_args()

# setup data provider
dataset = CIFAR10(path=args.data_dir,
                  normalize=False,
                  contrast_normalize=True,
                  whiten=True)
train = dataset.train_iter
test = dataset.valid_iter

init_uni = Uniform(low=-0.1, high=0.1)
opt_gdm = GradientDescentMomentum(learning_rate=0.01, momentum_coef=0.9)


# set up the model layers
bn = True
layers = [Conv((5, 5, 16), init=init_uni, activation=Rectlin(), batch_norm=bn),
          Affine(nout=500, init=init_uni, activation=Rectlin(), batch_norm=bn),
          Affine(nout=10, axes=Axes(ax.Y,), init=init_uni, activation=Softmax())]

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

model = Model(layers=layers)
callbacks = Callbacks(model, eval_set=test, **args.callback_args)
model.initialize(
    dataset=train,
    input_axes=Axes((ax.C, ax.H, ax.W)),
    target_axes=Axes((ax.Y,)),
    optimizer=opt_gdm,
    cost=cost,
    metric=Misclassification()
)

np.seterr(divide='raise', over='raise', invalid='raise')
model.fit(train, num_epochs=args.epochs, callbacks=callbacks)
print('Misclassification error = %.2f%%' % (model.eval(test) * 100))
