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
#
# Then run the example:
#
# test_dataloader.py --backend cpu

"""
The number of processed data should equal the number of test data.
"""
from __future__ import print_function

from neon.util.argparser import NeonArgparser
from neon.data import ImageLoader

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()
args.epochs = 1

# setup data provider
imgset_options = dict(inner_size=32, scale_range=40, aspect_ratio=110,
                      repo_dir=args.data_dir, subset_pct=args.subset_pct)
train = ImageLoader(set_name='train', shuffle=False,
                    do_transforms=False, **imgset_options)
test = ImageLoader(set_name='validation', shuffle=False,
                   do_transforms=False, **imgset_options)

# TODO: not all data are used for training and testing
nprocessed = 0
for x, t in test:
    nsteps = x.shape[
        1] // 128 if not isinstance(x, list) else x[0].shape[1] // 128
    nprocessed += 128
print("processed #test data: {:d}".format(nprocessed))
print("#test data: {:d}".format(test.ndata))
assert(nprocessed == test.ndata)
