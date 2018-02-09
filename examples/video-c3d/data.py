# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
from ngraph.frontends.neon.aeon_shim import AeonDataLoader
from ngraph.util.persist import get_data_cache_or_nothing

'''
Contains the helper functions for video_c3d.py
'''


def common_config(manifest_file, manifest_root, batch_size):
    '''
    Common configuration file for aeon loader

    manifest_file(str): Name of the manifest file
    manifest_root(str): Path for the manifest file
    batch_size(int): Batch size used for training
    '''

    cache_root = get_data_cache_or_nothing('ucf-cache/')
    video_config = {'type': "video",
                    'max_frame_count': 16,
                    'frame': {'height': 112, 'width': 112}}

    label_config = {'type': "label",
                    'binary': True}

    augmentation_config = {'type': 'image',
                           'scale': [0.875, 0.875]}

    configs = {'manifest_filename': manifest_file,
               'manifest_root': manifest_root,
               'batch_size': batch_size,
               'block_size': 5000,
               'augmentation': [augmentation_config],
               'cache_directory': cache_root,
               'etl': [video_config, label_config]}

    return configs


def make_validation_loader(manifest_file, manifest_root, batch_size, subset_pct=100):
    '''
    Validation data configuration for aeon loader. Returns the object to be used for getting
    validatin data.

    manifest_file(str): Name of the manifest file
    manifest_root(str): Path for the manifest file
    batch_size(int): Batch size used for training
    subset_pct(int): Percent data to be used for validation
    '''
    aeon_config = common_config(manifest_file, manifest_root, batch_size)
    aeon_config['subset_fraction'] = float(subset_pct / 100.0)

    dl = AeonDataLoader(aeon_config)

    return dl


def make_train_loader(manifest_file, manifest_root, batch_size, subset_pct=100, random_seed=0):
    '''
    Training data configuration for aeon loader. Returns the object to be used for getting
    training data.

    manifest_file(str): Name of the manifest file
    manifest_root(str): Path for the manifest file
    batch_size(int): Batch size used for training
    subset_pct(int): Percent data to be used for training
    random_seed(int): Random number generator seed
    '''
    aeon_config = common_config(manifest_file, manifest_root, batch_size)
    aeon_config['subset_fraction'] = float(subset_pct / 100.0)
    aeon_config['shuffle_manifest'] = True
    aeon_config['shuffle_enable'] = True
    aeon_config['random_seed'] = random_seed
    aeon_config['augmentation'][0]['center'] = False
    aeon_config['augmentation'][0]['flip_enable'] = True

    dl = AeonDataLoader(aeon_config)

    return dl
