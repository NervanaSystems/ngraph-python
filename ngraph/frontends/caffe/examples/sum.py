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
import ngraph.transformers as ngt
from ngraph.frontends.caffe.cf_importer.importer import parse_prototxt

model = "sum.prototxt"
# import graph from the prototxt
op_map = parse_prototxt(model, verbose=True)
# get the op handle for any layer
op = op_map.get("D")
# execute the op handle
res = ngt.make_transformer().computation(op)()
print("Result is:", res)
# EOF
