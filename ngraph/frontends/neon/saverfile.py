#!/usr/bin/env python
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
import os
import numpy as np


class SaverFile(object):
    def __init__(self, name):
        """
        A class that write and read dictionary of numpy.ndarray's with Op name as key to file

        Arguments:
            Name (string): Name of file used for saving.

        Methods:
            write_values: write dictionary of numpy.ndarray's with Op name as key to file
            read_values: read and return dictionary of numpy.ndrarry's with Op name as key
        """

        filename, fileext = os.path.splitext(name)
        if fileext is not "":
            assert fileext == ".npz"
        self.name = filename

    def write_values(self, tensors, compress):
        """
        write dictionary of numpy.ndarray's with Op name as key to file

        Arguments:
            tensors (dict): A dictionary of numpy.ndarray's with Op name as key
        """
        if compress:
            np.savez_compressed(self.name, **tensors)
        else:
            np.savez(self.name, **tensors)

    def read_values(self):
        """
        read and return dictionary of numpy.ndrarry's with Op name as key

        Returns:
            dictionary of numpy.ndrarry's with Op name as key
        """
        tensors = dict()
        filename = self.name + ".npz"
        with np.load(filename) as npzfile:
            for file in npzfile.files:
                tensors[file] = npzfile[file]
        return tensors
