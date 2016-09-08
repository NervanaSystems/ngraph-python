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

import geon.dataloader.dataloaderbackend
from geon.op_graph.arrayaxes import *
from geon.op_graph.convolution import convolution
from geon.op_graph.op_graph import *
from geon.transformers.nptransform import NumPyTransformer
from geon.util.names import *
try:
    from geon.transformers.argon.artransform import ArgonTransformer
except ImportError as e:
    if 'argon' in str(e):
        print("Argon backend and tensor are defined in argon package, which is not installed.")
    elif 'mpi4py' in str(e):
        print(
            "Argon backend currently depends on the package mpi4py, which is not installed."
        )
    else:
        raise
