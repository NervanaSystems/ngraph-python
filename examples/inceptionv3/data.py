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


def make_aeon_loaders(train_manifest, valid_manifest,
                      batch_size, train_iterations, datadir, random_seed=1,
                      dataset="i1k"):
    """
    datadir is the path for the images
    train_manifest is the name of tab separated file for AEON for training images
    valid_manifest is the name of tab separated file for AEON for validation images
    Both manifest files are assumed to be under datadir
    """
    if(dataset == "i1k"):
        train_manifest = datadir + "train-index-tabbed.csv"
        valid_manifest = datadir + "val-index-tabbed.csv"
    else:
        print("Only Imagenet 1K is supported")
        exit()

    def common_config(manifest_file, batch_size, dataset=dataset):
        if(dataset == "i1k"):
            cache_root = get_data_cache_or_nothing("i1k-cache/")

            image_config = {"type": "image",
                            "height": 299,
                            "width": 299}

            label_config = {"type": "label",
                            "binary": False}

            augmentation = {"type": "image",
                            "flip_enable": True}

            return {'manifest_filename': manifest_file,
                    'manifest_root': datadir,
                    'batch_size': batch_size,
                    'block_size': 5000,
                    'cache_directory': cache_root,
                    'etl': [image_config, label_config],
                    'augmentation': [augmentation]}
            print("Imagenet")
        else:
            raise NotImplementedError("Only Imagenet 1K is supported")

    train_config = common_config(train_manifest, batch_size)
    train_config['iteration_mode'] = "COUNT"
    train_config['iteration_mode_count'] = train_iterations
    train_config['shuffle_manifest'] = False
    train_config['shuffle_enable'] = False
    train_config['random_seed'] = random_seed

    valid_config = common_config(valid_manifest, batch_size)
    valid_config['iteration_mode'] = "ONCE"

    train_loader = AeonDataLoader(train_config)
    valid_loader = AeonDataLoader(valid_config)

    return (train_loader, valid_loader)
