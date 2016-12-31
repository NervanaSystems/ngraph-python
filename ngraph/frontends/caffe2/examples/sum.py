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
from ngraph.frontends.caffe2.c2_importer.importer import C2Importer
from caffe2.python import core, workspace
import ngraph.transformers as ngt
import numpy as np

# Caffe2 - network creation
net = core.Net("net")
shape = (2, 2, 2)

A = net.GivenTensorFill([], "A", shape=shape, values=np.random.uniform(-5, 5, shape), name="A")
B = net.GivenTensorFill([], "B", shape=shape, values=np.random.uniform(-5, 5, shape), name="B")
C = net.GivenTensorFill([], "C", shape=shape, values=np.random.uniform(-5, 5, shape), name="C")
Y = A.Sum([B, C], ["Y"], name="Y")

# Execute via Caffe2
workspace.ResetWorkspace()
workspace.RunNetOnce(net)

# Execute in numpy
a = workspace.FetchBlob("A")
b = workspace.FetchBlob("B")
c = workspace.FetchBlob("C")

np_result = np.sum([a, b, c], axis=0)

# Import caffe2 network into ngraph
importer = C2Importer()
importer.parse_net_def(net.Proto(), verbose=False)

# Get handle
f_ng = importer.get_op_handle("Y")

# Execute in ngraph
f_result = ngt.make_transformer().computation(f_ng)()

# compare numpy, Caffe2 and ngraph results
print("Caffe2 result: \n{}\n".format(workspace.FetchBlob("Y")))
print("ngraph result: \n{}\n".format(f_result))
print("numpy result: \n{}\n".format(np_result))

assert(np.allclose(f_result, workspace.FetchBlob("Y")))
assert(np.allclose(f_result, np_result))
