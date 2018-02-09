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
import os
from ngraph.util.persist import get_data_cache_or_nothing
from ngraph.frontends.neon.aeon_shim import AeonDataLoader
from ngraph.frontends.neon.data.lsun import LSUN


def make_aeon_loaders(work_dir, batch_size, train_iterations, random_seed=0, subset_pct=100.0):
    categories = LSUN.lsun_categories()  # assumes most recent tag
    lbl_map = dict(zip(categories, range(len(categories))))

    manifest_file = LSUN(path=work_dir).ingest_lsun(category='bedroom',
                                                    dset='train',
                                                    lbl_map=lbl_map,
                                                    tag='latest',
                                                    overwrite=False,
                                                    png_conv=True)

    def common_config(manifest_file, batch_size, subset_pct):
        cache_root = get_data_cache_or_nothing('lsun-cache/')

        image_config = {"type": "image",
                        "height": 64,
                        "width": 64}
        label_config = {"type": "label",
                        "binary": False}

        return {'manifest_filename': manifest_file,
                'manifest_root': os.path.dirname(manifest_file),
                'subset_fraction': float(subset_pct / 100.0),
                'batch_size': batch_size,
                'cache_directory': cache_root,
                'etl': [image_config, label_config]}

    aeon_config = common_config(manifest_file, batch_size, subset_pct=subset_pct)
    aeon_config['iteration_mode'] = "INFINITE"
    aeon_config['iteration_mode_count'] = train_iterations
    aeon_config['shuffle_manifest'] = True
    aeon_config['shuffle_enable'] = True
    aeon_config['random_seed'] = random_seed

    dataloader = AeonDataLoader(aeon_config)
    return dataloader
