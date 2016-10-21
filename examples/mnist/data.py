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
import numpy as np
import os
from tqdm import tqdm
from aeon import DataLoader


def ensure_dirs_exist(path):
    """
    Simple helper that ensures that any directories specified in the path are
    created prior to use.

    Arguments:
        path (str): the path (may be to a file or directory).  Any intermediate
                    directories will be created.

    Returns:
        str: The unmodified path value.
    """
    outdir = os.path.dirname(path)
    if outdir != '' and not os.path.isdir(outdir):
        os.makedirs(outdir)
    return path


def get_data_cache_or_nothing(subdir=''):
    cache_root = os.getenv("NEON_DATA_CACHE_DIR", '')

    return ensure_dirs_exist(os.path.join(cache_root, subdir))


class NpyBackend(object):
    def __init__(self):
        self.use_pinned_mem = False
        self.rng_seed = None

    def consume(self, buf_index, hostlist, devlist):
        assert 0 <= buf_index < 2, 'Can only double buffer'
        hb = np.rollaxis(hostlist[buf_index], 0, hostlist[buf_index].ndim)
        if devlist[buf_index] is None:
            devlist[buf_index] = np.empty_like(hb)
        devlist[buf_index][:] = hb


def common_config(manifest_file, batch_size):
    cache_root = get_data_cache_or_nothing('mnist-cache/')


    return {
               'manifest_filename': manifest_file,
               'minibatch_size': batch_size,
               'macrobatch_size': 25000,
               'cache_directory': cache_root,
               'type': 'image,label',
               'image': {'height': 28,
                         'width': 28,
                         'channels': 1},
               'label': {'binary': False}
            }


def make_aeon_loaders(train_manifest, valid_manifest, batch_size, random_seed=0):
    train_config = common_config(train_manifest, batch_size)
    # train_config['shuffle_manifest'] = True
    # train_config['shuffle_every_epoch'] = True
    # train_config['random_seed'] = random_seed
    # train_config['image']['center'] = False

    valid_config = common_config(valid_manifest, batch_size)

    backend = NpyBackend()
    train_loader = DataLoader(train_config, backend)
    valid_loader = DataLoader(valid_config, backend)

    return (train_loader, valid_loader)
