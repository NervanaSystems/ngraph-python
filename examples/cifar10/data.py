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
from PIL import Image
from tqdm import tqdm
from ngraph.frontends.neon.aeon_shim import AeonDataloader
from ngraph.util.persist import get_data_cache_or_nothing
from cifar10 import CIFAR10


def ingest_cifar10(root_dir, padded_size=32, overwrite=False):
    '''
    Save CIFAR-10 dataset as PNG files
    '''
    out_dir = os.path.join(root_dir, 'cifar10')

    set_names = ('train', 'valid')
    manifest_files = [os.path.join(out_dir, setn + '-index.csv') for setn in set_names]

    if (all([os.path.exists(manifest) for manifest in manifest_files]) and not overwrite):
        return manifest_files

    datasets = CIFAR10(out_dir).load_data()

    pad_size = (padded_size - 32) // 2 if padded_size > 32 else 0
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))

    # Now write out image files and manifests
    for setn, manifest, data in zip(set_names, manifest_files, datasets):
        records = [('@FILE', 'STRING')]
        img_path = os.path.join(out_dir, setn)
        if not os.path.isdir(img_path):
            os.makedirs(img_path)

        for idx, (img, lbl) in enumerate(tqdm(zip(data['image']['data'], data['label']['data']))):
            im = np.pad(img.reshape((3, 32, 32)), pad_width, mode='mean')
            im = Image.fromarray(np.uint8(np.transpose(im, axes=[1, 2, 0]).copy()))
            fname = os.path.join(img_path, '{}_{:05d}.png'.format(lbl, idx))
            im.save(fname, format='PNG')
            records.append((os.path.relpath(fname, out_dir), lbl))
        np.savetxt(manifest, records, fmt='%s\t%s')

    return manifest_files


def make_aeon_loaders(work_dir, batch_size, train_iterations, random_seed=0):
    train_manifest, valid_manifest = ingest_cifar10(work_dir, padded_size=40)

    def common_config(manifest_file, batch_size):
        cache_root = get_data_cache_or_nothing('cifar10-cache/')

        image_config = {"type": "image",
                        "height": 32,
                        "width": 32}
        label_config = {"type": "label",
                        "binary": False}
        augmentation = {"type": "image",
                        "scale": [0.8, 0.8]}

        return {'manifest_filename': manifest_file,
                'manifest_root': os.path.dirname(manifest_file),
                'batch_size': batch_size,
                'block_size': 5000,
                'cache_directory': cache_root,
                'etl': [image_config, label_config],
                'augmentation': [augmentation]}

    train_config = common_config(train_manifest, batch_size)
    train_config['iteration_mode'] = "COUNT"
    train_config['iteration_mode_count'] = train_iterations
    train_config['shuffle_manifest'] = True
    train_config['shuffle_enable'] = True
    train_config['random_seed'] = random_seed
    train_config['augmentation'][0]["center"] = False
    train_config['augmentation'][0]["flip_enable"] = True

    valid_config = common_config(valid_manifest, batch_size)
    valid_config['iteration_mode'] = "ONCE"

    import json
    train_loader = AeonDataloader(json.dumps(train_config))
    valid_loader = AeonDataloader(json.dumps(valid_config))

    return (train_loader, valid_loader)
