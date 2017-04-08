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
from ngraph.frontends.caffe.cf_importer.importer import CaffeImporter

model = "sum.prototxt"
#import graph from the prototxt
importer = CaffeImporter()
importer.parse_net_def(model,verbose=True)
#call the compute function and print results
res = importer.compute("A,B,C,D")
print("Result is:")
for out in res:
    print("\n",out)

