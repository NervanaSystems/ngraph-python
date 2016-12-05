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
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.neon import NgraphArgparser

import numpy as np

parser = NgraphArgparser(description='sum example')
args = parser.parse_args()
transformer_name = args.backend

# hard code flex transformer
#transformer_name = 'flexgpu'
#factory = ngt.make_transformer_factory(transformer_name)
#ngt.set_transformer_factory(factory)

transformer = ngt.make_transformer()

#nelems = 10
#H = ng.make_axis(length=nelems)
#x = ng.placeholder(H)

n, m = 2, 3
N = ng.make_axis(length=n)
M = ng.make_axis(length=m)
X = ng.placeholder((N, M))

z = abs(X)
comp = transformer.computation(z, X)

# input arg values
#Xval = np.arange(n*m).reshape(n,m).astype(np.float32)
Xval = np.array([5, 1, 0, -2, 3, 4]).reshape(n, m).astype(np.float32)
Xval[0,1] = -Xval[0,1]

print(Xval)
Z = comp(Xval)
print(Z)


if transformer_name == 'flexgpu':
    fm = transformer.flex_manager
    print(fm.stat_ids)

    # get maxabs
    fm.transfer_stats()
    print(fm.host_stats)
