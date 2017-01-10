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
from ngraph.util.persist import ensure_dirs_exist, pickle_load, valid_path_append, fetch_file
from PIL import Image
import tarfile


class CIFAR10(object):
    """
    CIFAR10 data set from https://www.cs.toronto.edu/~kriz/cifar.html

    Arguments:
        path (str): Local path to copy data files.
    """

    def __init__(self, path='.'):
        self.path = path
        self.url = 'http://www.cs.toronto.edu/~kriz'
        self.filename = "cifar-10-python.tar.gz"
        self.size = 170498071

    def load_data(self):
        """
        Fetch the CIFAR-10 dataset and load it into memory.

        Arguments:
            path (str, optional): Local directory in which to cache the raw
                                  dataset.  Defaults to current directory.
            normalize (bool, optional): Whether to scale values between 0 and 1.
                                        Defaults to True.

        Returns:
            tuple: Both training and test sets are returned.
        """
        workdir, filepath = valid_path_append(self.path, '', self.filename)
        if not os.path.exists(filepath):
            fetch_file(self.url, self.filename, filepath, self.size)

        batchdir = os.path.join(workdir, 'cifar-10-batches-py')
        if not os.path.exists(os.path.join(batchdir, 'data_batch_1')):
            assert os.path.exists(filepath), "Must have cifar-10-python.tar.gz"
            with tarfile.open(filepath, 'r:gz') as f:
                f.extractall(workdir)

        train_batches = [os.path.join(batchdir, 'data_batch_' + str(i)) for i in range(1, 6)]
        Xlist, ylist = [], []
        for batch in train_batches:
            with open(batch, 'rb') as f:
                d = pickle_load(f)
                Xlist.append(d['data'])
                ylist.append(d['labels'])

        X_train = np.vstack(Xlist).reshape(-1, 3, 32, 32)
        y_train = np.vstack(ylist).ravel()

        with open(os.path.join(batchdir, 'test_batch'), 'rb') as f:
            d = pickle_load(f)
            X_test, y_test = d['data'], d['labels']
            X_test = X_test.reshape(-1, 3, 32, 32)

        self.train_set = {'image': {'data': X_train,
                                    'axes': ('batch', 'channel', 'height', 'width')},
                          'label': {'data': y_train,
                                    'axes': ('batch',)}}
        self.valid_set = {'image': {'data': X_test,
                                    'axes': ('batch', 'channel', 'height', 'width')},
                          'label': {'data': np.array(y_test),
                                    'axes': ('batch',)}}

        return self.train_set, self.valid_set


def ingest_cifar10(root_dir, padded_size=32, overwrite=False):
    '''
    Save CIFAR-10 dataset as PNG files
    '''
    out_dir = os.path.join(root_dir, 'cifar10')
    set_names = ('train', 'val')
    manifest_files = [os.path.join(out_dir, setn + '-index.csv') for setn in set_names]

    if (all([os.path.exists(manifest) for manifest in manifest_files]) and not overwrite):
        return manifest_files

    dataset = {k: s for k, s in zip(set_names, CIFAR10(path=out_dir).load_data())}

    pad_size = (padded_size - 32) // 2 if padded_size > 32 else 0
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))

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
            img_path = os.path.join(img_paths[setn][lbl[0]], str(idx) + '.png')
            im = np.pad(img.reshape((3, 32, 32)), pad_width, mode='mean')
            im = Image.fromarray(np.uint8(np.transpose(im, axes=[1, 2, 0]).copy()))
            im.save(os.path.join(out_dir, img_path), format='PNG')
            records.append((os.path.relpath(img_path, out_dir),
                            os.path.relpath(lbl_paths[lbl[0]], out_dir)))
        np.savetxt(manifest, records, fmt='%s,%s')

    return manifest_files
