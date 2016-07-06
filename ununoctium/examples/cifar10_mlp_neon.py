#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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
# Prior to running, you need to write out padded cifar10 batches for ImageLoader to consume
#
# batch_writer.py --set_type cifar10 \
#       --data_dir <path-to-save-batches> \
#       --macro_size 10000 \
#       --target_size 40
#
# Then run the example:
#
# cifar10.py -r 0 -vv \
#      --log <logfile> \
#      --no_progress_bar \
#      --save_path <save-path> \
#      --eval_freq 1 \
#      --backend gpu \
#      --data_dir <path-to-saved-batches>
#
# This setting should get to ~6.7% top-1 error. (Could be as low as 6.5)
#
# NB:  It is good practice to set your data_dir where your batches are stored
# to be local to your machine (to avoid accessing the macrobatches over network if,
# for example, your data_dir is in an NFS mounted location)
"""
Small CIFAR10 based MLP with fully connected layers.
"""

from neon.initializers import Uniform
from neon.layers import GeneralizedCost, Affine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Misclassification, CrossEntropyMulti, Softmax, Rectlin
from neon.util.argparser import NeonArgparser
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()

# setup data provider
imgset_options = dict(inner_size=32, scale_range=40, aspect_ratio=110,
                      repo_dir=args.data_dir, subset_pct=args.subset_pct)
train = ImageLoader(set_name='train', shuffle=True, do_transforms=True, **imgset_options)
test = ImageLoader(set_name='validation', shuffle=False, do_transforms=False, **imgset_options)

init_uni0 = Uniform(low=-0.002 , high=0.002)
init_uni1 = Uniform(low=-0.1, high=0.1)
opt_gdm = GradientDescentMomentum(learning_rate=0.0001, momentum_coef=0.9)

# set up the model layers
layers = [Affine(nout=200, init=init_uni0, activation=Rectlin()),
          Affine(nout=10, init=init_uni1, activation=Softmax())]

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

mlp = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(mlp, eval_set=test, **args.callback_args)

mlp.fit(train, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

print('Misclassification error = %.1f%%' % (mlp.eval(test, metric=Misclassification())*100))
