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
import numpy as np
import random as random


def test_relu():
    workspace.ResetWorkspace()

    shape = [10, 10]
    data = [random.gauss(mu=0, sigma=10) for i in range(np.prod(shape))]

    net = core.Net("net")
    net.GivenTensorFill([], "X", shape=shape, values=data, name="X")
    net.Relu(["X"], ["Y"], name="Y")

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
    assert(np.array_equal(f_result, workspace.FetchBlob("Y")))


def test_tanh():
    workspace.ResetWorkspace()

    shape = [1, 10]
    data = [random.gauss(mu=0, sigma=10) for i in range(np.prod(shape))]

    net = core.Net("net")
    net.GivenTensorFill([], "X", shape=shape, values=data, name="X")
    net.Tanh(["X"], ["Y"], name="Y")

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
    assert(np.allclose(f_result, workspace.FetchBlob("Y"), atol=1e-4, rtol=0, equal_nan=False))


def test_softmax():
    workspace.ResetWorkspace()

    shape = [2, 7]
    data = [
        1., 2., 3., 4., 1., 2., 3.,
        1., 2., 3., 4., 1., 2., 3.
    ]
    expected = [
        [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175],
        [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175],
    ]

    net = core.Net("net")
    net.GivenTensorFill([], "X", shape=shape, values=data, name="X")
    net.Softmax(["X"], ["Y"], name="Y")

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

    # Get Caffe2 result
    c2_y = workspace.FetchBlob("Y")

    # compare Caffe2 and ngraph results
    assert(np.allclose(f_result, c2_y, atol=1e-4, rtol=0, equal_nan=False))

    # compare expected results and ngraph results
    assert(np.allclose(f_result, expected, atol=1e-3, rtol=0, equal_nan=False))


def test_exp():
    workspace.ResetWorkspace()

    shape = [2, 7]
    data = [
        1., 2., 3., 4., 1., 2., 3.,
        1., 2., 3., 4., 1., 2., 3.
    ]
    expected = [
        [2.71828, 7.3890, 20.08553, 54.59815, 2.71828, 7.3890, 20.08553],
        [2.71828, 7.3890, 20.08553, 54.59815, 2.71828, 7.3890, 20.08553],
    ]

    net = core.Net("net")
    net.GivenTensorFill([], "X", shape=shape, values=data, name="X")
    net.Exp(["X"], ["Y"], name="Y")

    # Execute via Caffe2
    workspace.RunNetOnce(net)

    # Import caffe2 network into ngraph
    importer = C2Importer()
    importer.parse_net_def(net.Proto(), verbose=False)

    # Get handle
    f_ng = importer.get_op_handle("Y")

    # Execute in ngraph and get the result
    ex = ExecutorFactory()
    f_result = ex.executor(f_ng)()

    # Get Caffe2 result
    c2_y = workspace.FetchBlob("Y")

    # compare Caffe2 and ngraph results
    assert(np.allclose(f_result, c2_y, atol=1e-4, rtol=0, equal_nan=False))

    # compare expected results and ngraph results
    assert(np.allclose(f_result, expected, atol=1e-3, rtol=0, equal_nan=False))


def test_NCHW2NHWC():
    workspace.ResetWorkspace()

    # NCHW
    shape = [2, 3, 4, 5]
    data1 = [float(i) for i in range(np.prod(shape))]

    net = core.Net("net")
    X = net.GivenTensorFill([], ["X"], shape=shape, values=data1, name="X")
    X.NCHW2NHWC([], ["Y"], name="Y")

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
    assert(np.array_equal(f_result, workspace.FetchBlob("Y")))


def test_NHWC2NCHW():
    workspace.ResetWorkspace()

    # NHWC
    shape = [2, 3, 4, 5]
    data1 = [float(i) for i in range(np.prod(shape))]

    net = core.Net("net")
    X = net.GivenTensorFill([], ["X"], shape=shape, values=data1, name="X")
    X.NCHW2NHWC([], ["Y"], name="Y")

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
    assert(np.array_equal(f_result, workspace.FetchBlob("Y")))