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
import os
import numpy as np
from aeon import DataLoader
from ngraph.util.persist import get_data_cache_or_nothing


def common_config(manifest_file, manifest_root, batch_size):
    cache_root = get_data_cache_or_nothing('cifar10-cache/')

    return {
               'manifest_filename': manifest_file,
               'manifest_root': manifest_root,
               'minibatch_size': batch_size,
               'macrobatch_size': 5000,
               'type': 'image,label',
               'cache_directory': cache_root,
               'image': {'height': 32,
                         'width': 32,
                         'scale': [0.8, 0.8]},
               'label': {'binary': False}
            }


def make_aeon_loaders(train_manifest, valid_manifest, batch_size, backend, random_seed=0):
    manifest_root = os.path.dirname(train_manifest)
    train_config = common_config(train_manifest, manifest_root, batch_size)
    # train_config['shuffle_manifest'] = True
    # train_config['shuffle_every_epoch'] = True
    # train_config['random_seed'] = random_seed
    # train_config['image']['center'] = False
    # train_config['image']['flip_enable'] = True

    valid_config = common_config(valid_manifest, manifest_root, batch_size)

    train_loader = DataLoader(train_config, backend)
    valid_loader = DataLoader(valid_config, backend)

    return (train_loader, valid_loader)

