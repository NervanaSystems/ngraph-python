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
from ngraph.testing import ExecutorFactory
import ngraph.transformers as ngt
import numpy as np
import random as random


def test_constant():
    workspace.ResetWorkspace()

    shape = [10, 10]
    val = random.random()
    net = core.Net("net")
    net.ConstantFill([], ["Y"], shape=shape, value=val, run_once=0, name="Y")

    # Execute via Caffe2
    workspace.RunNetOnce(net)

    # Import caffe2 network into ngraph
    importer = C2Importer()
    importer.parse_net_def(net.Proto(), verbose=False)

    # Get handle
    f_ng = importer.get_op_handle("Y")

    # Execute
    f_result = ngt.make_transformer().computation(f_ng)()

    # compare Caffe2 and ngraph results
    assert(np.ma.allequal(f_result, workspace.FetchBlob("Y")))
    assert(np.isclose(f_result[0][0], val, atol=1e-6, rtol=0))


def test_gausianfill():
    workspace.ResetWorkspace()

    shape = [10, 10]
    net = core.Net("net")
    net.GaussianFill([], ["Y"], shape=shape, mean=0.0, std=1.0, name="Y")

    # Execute via Caffe2
    workspace.RunNetOnce(net)

    # Import caffe2 network into ngraph
    importer = C2Importer()
    importer.parse_net_def(net.Proto(), verbose=False)

    # Get handle
    f_ng = importer.get_op_handle("Y")

    # Execute
    # f_result =
    ngt.make_transformer().computation(f_ng)()

    # print("Caffe2 result: \n{}\n".format(workspace.FetchBlob("Y")))
    # print("ngraph result: \n{}\n".format(f_result))

    # TODO: how to check it? Meybe we can omit this test
    pass


def test_uniformfill():
    workspace.ResetWorkspace()

    shape = [10, 10]
    net = core.Net("net")
    net.UniformFill([], ["Y"], shape=shape, min=-2., max=2., name="Y")

    # Execute via Caffe2
    workspace.RunNetOnce(net)

    # Import caffe2 network into ngraph
    importer = C2Importer()
    importer.parse_net_def(net.Proto(), verbose=False)

    # Get handle
    f_ng = importer.get_op_handle("Y")

    # Execute
    ex = ExecutorFactory()
    # f_result =
    ex.executor(f_ng)()

    # print("Caffe2 result: \n{}\n".format(workspace.FetchBlob("Y")))
    # print("ngraph result: \n{}\n".format(f_result))

    # TODO: how to check it? Meybe we can omit this test
    pass


def test_uniformintfill():
    workspace.ResetWorkspace()

    shape = [10, 10]
    net = core.Net("net")
    net.UniformIntFill([], ["Y"], shape=shape, min=-2, max=2, name="Y")

    # Execute via Caffe2
    workspace.RunNetOnce(net)

    # Import caffe2 network into ngraph
    importer = C2Importer()
    importer.parse_net_def(net.Proto(), verbose=False)

    # Get handle
    f_ng = importer.get_op_handle("Y")

    # Execute
    ex = ExecutorFactory()
    # f_result =
    ex.executor(f_ng)()

    # print("Caffe2 result: \n{}\n".format(workspace.FetchBlob("Y")))
    # print("ngraph result: \n{}\n".format(f_result))

    # TODO: how to check it? Meybe we can omit this test
    pass


def test_xavierfill():
    workspace.ResetWorkspace()

    shape = [10, 10]
    net = core.Net("net")
    net.XavierFill([], ["Y"], shape=shape, name="Y")

    # Execute via Caffe2
    workspace.RunNetOnce(net)

    # Import caffe2 network into ngraph
    importer = C2Importer()
    importer.parse_net_def(net.Proto(), verbose=False)

    # Get handle
    f_ng = importer.get_op_handle("Y")

    # Execute
    ex = ExecutorFactory()
    # f_result =
    ex.executor(f_ng)()

    # print("Caffe2 result: \n{}\n".format(workspace.FetchBlob("Y")))
    # print("ngraph result: \n{}\n".format(f_result))

    # TODO: how to check it? Meybe we can omit this test
    pass


def test_giventensorfill():
    workspace.ResetWorkspace()

    shape = [10, 10]
    data1 = np.random.random(shape)

    net = core.Net("net")
    net.GivenTensorFill([], ["Y"], shape=shape, values=data1, name="Y")

    # Execute via Caffe2
    workspace.RunNetOnce(net)

    # Import caffe2 network into ngraph
    importer = C2Importer()
    importer.parse_net_def(net.Proto(), verbose=False)

    # Get handle
    f_ng = importer.get_op_handle("Y")

    # Execute
    ex = ExecutorFactory()
    f_result = ex.executor(f_ng)()

    # compare Caffe2 and ngraph results
    assert(np.ma.allequal(f_result, workspace.FetchBlob("Y")))
    assert(np.ma.allclose(f_result, data1, atol=1e-6, rtol=0))
