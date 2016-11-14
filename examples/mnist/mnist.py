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
import gzip
from ngraph.util.persist import ensure_dirs_exist, pickle_load, valid_path_append, fetch_file
import os
from tqdm import tqdm
import numpy as np
from PIL import Image


class MNIST(object):
    """
    Arguments:
        path (str): Local path to copy data files.
    """
    def __init__(self, path='.'):
        self.path = path
        self.url = 'https://s3.amazonaws.com/img-datasets'
        self.filename = 'mnist.pkl.gz'
        self.size = 15296311

    def load_data(self):
        """
        Fetch the MNIST dataset and load it into memory.

        Arguments:
            path (str, optional): Local directory in which to cache the raw
                                  dataset.  Defaults to current directory.

        Returns:
            tuple: Both training and test sets are returned.
        """
        workdir, filepath = valid_path_append(self.path, '', self.filename)
        if not os.path.exists(filepath):
            fetch_file(self.url, self.filename, filepath, self.size)

        with gzip.open(filepath, 'rb') as f:
            self.train_set, self.valid_set = pickle_load(f)

        self.train_set = {'image': self.train_set[0].reshape(60000, 1, 28, 28),
                          'label': self.train_set[1]}
        self.valid_set = {'image': self.valid_set[0].reshape(10000, 1, 28, 28),
                          'label': self.valid_set[1]}

        return self.train_set, self.valid_set


def ingest_mnist(root_dir, overwrite=False):
    '''
    Save MNIST dataset as PNG files
    '''
    out_dir = os.path.join(root_dir, 'mnist')

    set_names = ('train', 'val')
    manifest_files = [os.path.join(out_dir, setn + '-index.csv') for setn in set_names]

    if (all([os.path.exists(manifest) for manifest in manifest_files]) and not overwrite):
        return manifest_files

    dataset = {k: s for k, s in zip(set_names, MNIST(out_dir, False).load_data())}

    # Write out label files and setup directory structure
    lbl_paths, img_paths = dict(), dict(train=dict(), val=dict())
    for lbl in range(10):
        lbl_paths[lbl] = ensure_dirs_exist(os.path.join(out_dir, 'labels', str(lbl) + '.txt'))
        np.savetxt(lbl_paths[lbl], [lbl], fmt='%d')
        for setn in ('train', 'val'):
            img_paths[setn][lbl] = ensure_dirs_exist(os.path.join(out_dir, setn, str(lbl) + '/'))

    # Now write out image files and manifests
    for setn, manifest in zip(set_names, manifest_files):
        records = []
        for idx, (img, lbl) in enumerate(tqdm(zip(*dataset[setn]))):
            img_path = os.path.join(img_paths[setn][lbl], str(idx) + '.png')
            im = Image.fromarray(img)
            im.save(os.path.join(out_dir, img_path), format='PNG')
            records.append((os.path.relpath(img_path, out_dir),
                            os.path.relpath(lbl_paths[lbl], out_dir)))
        np.savetxt(manifest, records, fmt='%s,%s')

    return manifest_files
