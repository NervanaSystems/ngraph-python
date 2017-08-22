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
"""
Test the walkthrough ipython notebooks
"""

import pytest
import nbformat
import glob
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

# grab notebooks
base_dir = os.path.dirname(__file__)
notebook_dir = os.path.join(base_dir, '../examples/walk_through/')
notebooks = sorted(glob.glob(os.path.join(notebook_dir, '*.ipynb')))

# grab python version for notebook processor
kernel = 'python{}'.format(sys.version_info[0])


@pytest.mark.parametrize("notebook", notebooks)
def test_notebooks(notebook):
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name=kernel)
        ep.preprocess(nb, {'metadata': {'path': notebook_dir}})
