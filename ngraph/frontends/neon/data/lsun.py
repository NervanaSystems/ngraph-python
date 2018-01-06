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
from __future__ import print_function
import numpy as np
import os
import shutil
from tqdm import tqdm
from PIL import Image
import subprocess
import zipfile
from ngraph.util.persist import valid_path_append, ensure_dirs_exist
import requests
try:
    import lmdb
except ImportError as e:
    print("Dependency not installed")
    raise(e)


MAP_SIZE = 1099511627776
MAX_NUM_INGEST_PROC = 100


class LSUN(object):
    """
    LSUN data set
    Arguments:
        path (str): Local path to copy data files.
    """

    url = 'http://lsun.cs.princeton.edu/htbin/'

    def __init__(self, path='.'):
        self.path = path

    @staticmethod
    def lsun_categories(tag='latest'):
        """
        Query LSUN_URL and return a list of LSUN categories
        Argument:
            tag (str): version tag, use "latest" for most recent
        """
        lsunurl = LSUN.url + 'list.cgi?tag=' + tag
        # return json.loads(f.read())
        return requests.get(lsunurl).json()

    def download_lsun(self, category, dset, tag='latest', overwrite=False):
        """
        Download LSUN data and unpack
        Arguments:
            category (str): LSUN category (valid selections: lsun_categories)
            dset (str): dataset, "train", "val", or "test"
            tag (str, optional): version tag, defaults to most recent
            overwrite (bool): whether to overwrite existing data
        """
        dfile = 'test_lmdb' if dset == 'test' else '{0}_{1}_lmdb'.format(category, dset)
        self.filepath = filepath = valid_path_append(self.path, dfile)
        if not os.path.exists(filepath) or overwrite:
            filepath += '.zip'
            if not os.path.exists(filepath):
                url = LSUN.url + \
                    'download.cgi?tag={0}&category={1}&set={2}'.format(tag, category, dset)
                print('Data download might take a long time.')
                print('Downloading {0} {1} set...'.format(category, dset))
                subprocess.call(['curl', url, '-o', filepath])
                # TODO
                # should change to fetch_file,
                # but currently did not get the correct "Content-length" or total_size
                # fetch_file(url, 'bedroom_train_lmdb.zip', filepath)
            print('Extracting {0} {1} set...'.format(category, dset))
            zf = zipfile.ZipFile(filepath, 'r')
            zf.extractall(self.path)
            zf.close()
            print('Deleting {} ...'.format(filepath))
            os.remove(filepath)
        else:
            pass  # data already downloaded
        print("LSUN {0} {1} dataset downloaded and unpacked.".format(category, dset))

    def ingest_lsun(self, category, dset, lbl_map, tag='latest', overwrite=False, png_conv=False):
        """
        Save LSUN dataset as WEBP or PNG files and generate config and log files
        Arguments:
            category (str): LSUN category
            dset (str): dataset, "train", "val", or "test"
            lbl_map (dict(str:int)): maps a category to an integer
            overwrite (bool): whether to overwrite existing data
            png_conv (bool): whether to convert to PNG images
        """
        self.download_lsun(category, dset, tag=tag, overwrite=overwrite)

        dpath = 'test' if dset == 'test' else '{0}_{1}'.format(category, dset)
        dpath = os.path.join(self.path, dpath)
        manifest_file = '{}_index.csv'.format(dpath)

        if os.path.exists(manifest_file) and not overwrite:
            print("LSUN {0} {1} dataset ingested.".format(category, dset))
            print("Manifest file is: " + manifest_file)
            return manifest_file
        if os.path.exists(dpath):
            shutil.rmtree(dpath)
        if os.path.exists(manifest_file):
            os.remove(manifest_file)
        os.makedirs(dpath)

        lbl_paths = dict()
        for lbl in lbl_map:
            lbl_paths[lbl] = ensure_dirs_exist(os.path.join(self.path, 'labels', lbl + '.txt'))
            np.savetxt(lbl_paths[lbl], [lbl_map[lbl]], fmt='%d')

        print('Exporting images...')
        env = lmdb.open(dpath + '_lmdb', map_size=MAP_SIZE,
                        max_readers=MAX_NUM_INGEST_PROC, readonly=True)
        count, records = 0, []
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, val in tqdm(cursor):
                image_out_path = os.path.join(dpath, key + '.webp')
                with open(image_out_path, 'w') as fp:
                    fp.write(val)
                count += 1
                if png_conv:  # in case WEBP is not supported, extra step of conversion to PNG
                    image_out_path_ = image_out_path
                    image_out_path = os.path.join(dpath, key + '.png')
                    im = Image.open(image_out_path_).convert('RGB')
                    im.save(image_out_path, 'png')
                    os.remove(image_out_path_)
                records.append((os.path.relpath(image_out_path, self.path),
                                os.path.relpath(lbl_paths[category], self.path)))
            np.savetxt(manifest_file, records, fmt='%s\t%s')
        print("LSUN {0} {1} dataset ingested.".format(category, dset))
        print("Manifest file is: " + manifest_file)
        return manifest_file
