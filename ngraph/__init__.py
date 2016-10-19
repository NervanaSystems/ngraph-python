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

import ngraph.dataloader.dataloaderbackend
from ngraph.op_graph.axes_ops import dimshuffle
from ngraph.op_graph.axes import *
from ngraph.op_graph.convolution import fprop_conv, fprop_pool, ConvolutionAxis
from ngraph.op_graph.debug import PrintOp
from ngraph.op_graph.op_graph import *
from ngraph.transformers.nptransform import NumPyTransformer
from ngraph.util.names import *

try:
    from ngraph.transformers.gputransform import GPUTransformer
except ImportError:
    pass

try:
    from ngraph.transformers.argon.artransform import ArgonTransformer
except ImportError as e:
    if 'argon' in str(e):
        print("Argon backend and tensor are defined in argon package, which is not installed.")
    elif 'mpi4py' in str(e):
        print(
            "Argon backend currently depends on the package mpi4py, which is not installed."
        )
    else:
        raise
