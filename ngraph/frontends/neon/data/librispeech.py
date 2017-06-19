# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
import glob
import fnmatch
import tarfile
import contextlib
from tqdm import tqdm
from ngraph.util.persist import valid_path_append, fetch_file


VERSION_URLS = {"train-clean-100": "train-clean-100.tar.gz",
                "train-clean-360": "train-clean-360.tar.gz",
                "train-clean-500": "train-other-500.tar.gz",
                "dev-clean": "dev-clean.tar.gz",
                "dev-other": "dev-other.tar.gz",
                "test-clean": "test-clean.tar.gz",
                "test-other": "test-other.tar.gz"}


class Librispeech(object):
    """
    Librispeech data set from http://www.openslr.org/resources/12

    Arguments:
        manifest_file (str): Path to existing manifest file or
                             desired output manifest file
        path (str): Data directory to find the data. If it doesn't exist,
                    it will be downloaded.
        version (str): Dataset version - one of "train-clean-100", "train-clean-360",
                       "train-clean-500", "dev-clean", "dev-other",
                       "test-clean", or "test-other"
    """

    url = "http://www.openslr.org/resources/12"

    def __init__(self, manifest_file=None, path='.', version="dev-clean"):

        self.path = path
        if version not in VERSION_URLS:
            poss_versions = ", ".join(sorted(list(VERSION_URLS.keys())))
            raise ValueError(("{} is not a known Librispeech version. "
                              "Possible versions are: {}.").format(version,
                                                                   poss_versions))
        self.source_file = VERSION_URLS[version]
        self.version = version
        self.manifest_file = manifest_file

    def load_data(self, data_directory=None, manifest_file=None):
        """
        Create a manifest file for the requested dataset. First downloads the
        dataset and extracts it, if necessary.

        Arguments:
            data_directory (str): Path to data directory. Defaults to <path>/<version>
            manifest_file (str): Path to manifest file. Defaults to <data_directory>/manifest.tsv

        Returns:
            Path to manifest file
        """

        if manifest_file is None:
            if self.manifest_file is not None:
                manifest_file = self.manifest_file
            else:
                manifest_file = os.path.join(self.path, "manifest.tsv")

        if os.path.exists(manifest_file):
                return manifest_file

        # Download the file
        workdir, filepath = valid_path_append(self.path, '', self.source_file)
        if not os.path.exists(filepath):
            fetch_file(self.url, self.source_file, filepath)

        # Untar the file
        if data_directory is None:
            data_directory = os.path.join(self.path, self.version)
        if not os.path.exists(data_directory):
            print("Extracting tar file to {}".format(data_directory))
            with contextlib.closing(tarfile.open(filepath)) as tf:
                tf.extractall(data_directory)

        # Ingest the file
        ingest_librispeech(data_directory, manifest_file)

        return manifest_file


def get_files(directory, pattern, recursive=True):
    """ Return the full path to all files in directory matching the specified
    pattern.

    Arguments:
        directory (str): Directory path in which to look
        pattern (str): A glob pattern for filenames
        recursive (bool): Searches recursively if True

    Returns:
        A list of matching file paths
    """

    # This yields an iterator which really speeds up looking through large, flat directories
    if recursive is False:
        it = glob.iglob(os.path.join(directory, pattern))
        return it

    # If we want to recurse, use os.walk instead
    matches = list()
    for root, dirnames, filenames in os.walk(directory):
        matches.extend([os.path.join(root, ss) for ss in
                        fnmatch.filter(filenames, pattern)])

    return matches


def ingest_librispeech(input_directory, manifest_file=None, absolute_paths=True):
    """ Finds all .txt files and their indicated .flac files and writes them to an Aeon
    compatible manifest file.

    Arguments:
        input_directory (str): Path to librispeech directory
        manifest_file (str): Path to manifest file to output.
        absolute_paths (bool): Whether audio file paths should be absolute or
                               relative to input_directory.
    """

    if not os.path.isdir(input_directory):
        raise IOError("Data directory does not exist! {}".format(input_directory))

    if manifest_file is None:
        manifest_file = os.path.join(input_directory, manifest_file)

    transcript_files = get_files(input_directory, pattern="*.txt")
    if len(transcript_files) == 0:
        raise IOError("No .txt files were found in {}".format(input_directory))

    tqdm.write("Preparing manifest file...")
    with open(manifest_file, "w") as manifest:
        manifest.write("@FILE\tSTRING\n")
        for tfile in tqdm(transcript_files, unit=" Files", mininterval=.001):
            directory = os.path.dirname(tfile)
            if absolute_paths is False:
                directory = os.path.relpath(directory, input_directory)

            with open(tfile, "r") as fid:
                for line in fid.readlines():
                    id_, transcript = line.split(" ", 1)
                    afile = "{}.flac".format(os.path.join(directory, id_))
                    manifest.write("{}\t{}\n".format(afile, transcript))
