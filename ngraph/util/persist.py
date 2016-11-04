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
import os
import sys
import requests
from tqdm import tqdm

PY3 = sys.version_info[0] >= 3

if not PY3:
    import cPickle as the_pickle  # noqa
else:
    import pickle as the_pickle  # noqa

pickle = the_pickle


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
    cache_root = os.getenv("NGRAPH_DATA_CACHE_DIR", '')
    return '' if cache_root == '' else ensure_dirs_exist(os.path.join(cache_root, subdir))


def valid_path_append(path, *args):
    """
    Helper to validate passed path directory and append any subsequent
    filename arguments.

    Arguments:
        path (str): Initial filesystem path.  Should expand to a valid
                    directory.
        *args (list, optional): Any filename or path suffices to append to path
                                for returning.

        Returns:
            (list, str): path prepended list of files from args, or path alone if
                     no args specified.

    Raises:
        ValueError: if path is not a valid directory on this filesystem.
    """
    full_path = os.path.expanduser(path)
    res = []
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    if not os.path.isdir(full_path):
        raise ValueError("path: {0} is not a valid directory".format(path))
    for suffix_path in args:
        res.append(os.path.join(full_path, suffix_path))
    if len(res) == 0:
        return path
    elif len(res) == 1:
        return res[0]
    else:
        return res


def pickle_load(filepath):
    """
    Py2Py3 compatible Pickle load

    Arguments:
        filepath (str): File containing pickle data stream to load

    Returns:
        Unpickled object
    """
    if PY3:
        return pickle.load(filepath, encoding='latin1')
    else:
        return pickle.load(filepath)


def fetch_file(url, sourcefile, destfile, totalsz):
    """
    Download the file specified by the given URL.

    Args:
        url (str): Base URL of the file to be downloaded.
        sourcefile (str): Name of the source file.
        destfile (str): Path to the destination.
        totalsz (int): Size of the file to be downloaded.
    """
    req = requests.get(os.path.join(url, sourcefile),
                       headers={'User-Agent': 'ngraph'},
                       stream=True)

    chunksz = 1024**2
    nchunks = totalsz // chunksz

    print("Downloading file to: {}".format(destfile))
    with open(destfile, 'wb') as f:
        for data in tqdm(req.iter_content(chunksz), total=nchunks):
            f.write(data)
    print("Download Complete")
