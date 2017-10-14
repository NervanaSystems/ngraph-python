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
from __future__ import division, print_function, absolute_import
import six
from contextlib import contextmanager
from zipfile import ZipFile

class SaverFile(object):
    def __init__(self, Name="Weights"):
        self.Name = Name
        super(SaverFile, self).__init__(**kwargs)
    
    def write_values(self, values):
        with ZipFile(self.name, 'w') as zf:
            pass

    def read_values(self):
        with ZipFile(self.name, 'r') as zf:
            pass

        
