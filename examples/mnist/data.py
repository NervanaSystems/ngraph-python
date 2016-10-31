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
from mnist import ingest_mnist

def common_config(manifest_file, batch_size):
    cache_root = get_data_cache_or_nothing('mnist-cache/')

    return {
               'manifest_filename': manifest_file,
               'manifest_root': os.path.dirname(manifest_file),
               'minibatch_size': batch_size,
               'macrobatch_size': 25000,
               'cache_directory': cache_root,
               'type': 'image,label',
               'image': {'height': 28,
                         'width': 28,
                         'channels': 1},
               'label': {'binary': False}
            }


def make_aeon_loaders(work_dir, batch_size, backend, random_seed=0):
    train_manifest, valid_manifest = ingest_mnist(work_dir)

    train_config = common_config(train_manifest, batch_size)
    # train_config['shuffle_manifest'] = True
    # train_config['shuffle_every_epoch'] = True
    # train_config['random_seed'] = random_seed
    # train_config['image']['center'] = False

    valid_config = common_config(valid_manifest, batch_size)

    train_loader = DataLoader(train_config, backend)
    valid_loader = DataLoader(valid_config, backend)

    return (train_loader, valid_loader)
