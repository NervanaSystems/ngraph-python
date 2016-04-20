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
# Prior to running, you need to write out padded cifar10 batches for ImageLoader to consume
#
# batch_writer.py --set_type cifar10 \
#       --data_dir <path-to-save-batches> \
#       --macro_size 10000 \
#       --target_size 40
#

import geon.backends.graph.graph as graph

from neon.util.argparser import NeonArgparser
from neon.data import ImageLoader

import geon.backends.graph.funs as be

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()
# setup data provider
imgset_options = dict(inner_size=32, scale_range=40, aspect_ratio=110,
                      repo_dir=args.data_dir, subset_pct=args.subset_pct)


class cifar_mlp(object):
    def __init__(self, nclasses=10, height=32, width=32, colors=3, **kargs):
        super(cifar_mlp, self).__init__(**kargs)

        # Data shape
        self.h = be.axis(height)
        self.w = be.axis(width)
        self.c = be.axis(colors)

        # Result shape
        self.nclasses = be.axis(nclasses)

        # Hidden layer
        self.h0 = be.axis(200)

        # Result
        self.nclasses = be.axis(nclasses)

        # Params to be trained
        self.w0 = be.array(shape=(self.h, self.w, self.c, self.h0))
        self.w1 = be.array(shape=(self.h0, self.nclasses))

    # Given an description of an input, return the description of the output
    def fprop(self, x):
        # This is a declaration about the input

        x = be.array(array=x, shape=(self.h, self.w, self.c))
        return be.logistic(be.dot(self.w1, be.relu(be.dot(self.w0, x))))

    def train(self):
        # Here we need to instantiate for a backend

        self.w0.initialize(init=be.Uniform(low=-0.004, high=0.004))
        self.w1.initialize(init=be.Uniform(low=-0.1, high=0.1))

        train = ImageLoader(set_name='train', shuffle=True, do_transforms=True, **imgset_options)

        be.fit(self,
               train,
               cost=be.CrossEntropyBinary(),
               optimizer=be.GradientDescentMomentum(learning_rate=0.0002, momentum_coef=0.9),
               num_epochs=args.epochs)

    def eval(self):
        test = ImageLoader(set_name='validation', shuffle=False, do_transforms=False, **imgset_options)

        






