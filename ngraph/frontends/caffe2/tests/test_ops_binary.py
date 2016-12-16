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
from caffe2.python import core, workspace
from ngraph.frontends.caffe2.c2_importer.importer import C2Importer
import ngraph.transformers as ngt
import numpy as np
import pytest

#TODO: how to use the same data for both transformers
@pytest.mark.xfail(strict=True, reason="c2 and ngraph generates own data")
def test_sum():
    net = core.Net("net")
    X = net.GaussianFill([], ["X"], shape=[2, 2], mean=0.0, std=1.0, run_once=0, name="X")
    W = net.GaussianFill([], ["W"], shape=[2, 2], mean=0.0, std=1.0, run_once=0, name="W")
    b = net.ConstantFill([], ["b"], shape=[2, ], value=1.0, run_once=0, name="b")
    Y = X.FC([W, b], ["Y"], name="Y")

    # Execute via Caffe2
    workspace.ResetWorkspace()
    workspace.RunNetOnce(net)

    # Import caffe2 network into ngraph
    importer = C2Importer()
    importer.parse_net_def(net.Proto(), verbose=False)

    # Get handle
    f_ng = importer.get_op_handle("Y")

    # Execute
    f_result = ngt.make_transformer().computation(f_ng)()

    # compare Caffe2 and ngraph results
    assert( np.array_equal(f_result, workspace.FetchBlob("Y")))
