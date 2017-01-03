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


def test_fc():
    workspace.ResetWorkspace()

    shape = [10, 10]
    data1 = [random.gauss(mu=0, sigma=10) for i in range(np.prod(shape))]
    data2 = [random.gauss(mu=0, sigma=10) for i in range(np.prod(shape))]

    net = core.Net("net")
    X = net.GivenTensorFill([], ["X"], shape=shape, values=data1, name="X")
    W = net.GivenTensorFill([], ["W"], shape=shape, values=data2, name="W")
    b = net.ConstantFill([], ["b"], shape=[shape[0]], value=1.0, run_once=0, name="b")
    net.FC([X, W, b], ["Y"], name="Y")

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
    assert(np.allclose(f_result, workspace.FetchBlob("Y"), atol=1e-4, rtol=1e-3, equal_nan=False))


def test_AveragedLoss():
    workspace.ResetWorkspace()
    shape = (32,)

    net = core.Net("net")
    X = net.GivenTensorFill([], "Y", shape=shape, values=np.random.uniform(-1, 1, shape))
    X.AveragedLoss([], ["loss"])

    # Execute via Caffe2
    workspace.RunNetOnce(net)

    # Import caffe2 network into ngraph
    importer = C2Importer()
    importer.parse_net_def(net.Proto(), verbose=False)

    # Get handle
    f_ng = importer.get_op_handle("loss")

    # Execute
    ex = ExecutorFactory()
    f_result = ex.executor(f_ng)()

    assert(np.allclose(f_result, workspace.FetchBlob("loss"), equal_nan=False))


def test_LabelCrossEntropy():
    workspace.ResetWorkspace()
    batch = 8
    classes = 16
    y_shape = (batch, classes)
    t_shape = (batch, )
    y_values = np.random.uniform(0, 1, y_shape)
    t_values = np.random.randint(0, classes, t_shape)

    net = core.Net("net")
    Y = net.GivenTensorFill([], "Y", shape=y_shape, values=y_values)
    T = net.GivenTensorIntFill([], "T", shape=t_shape, values=t_values)
    net.LabelCrossEntropy([Y, T], "xent")

    # Execute via Caffe2
    workspace.RunNetOnce(net)

    # Import caffe2 network into ngraph
    importer = C2Importer()
    importer.parse_net_def(net.Proto(), verbose=True)

    # Get handle
    f_ng = importer.get_op_handle("xent")

    # Execute
    ex = ExecutorFactory()
    f_result = ex.executor(f_ng)()

    assert(np.allclose(f_result, workspace.FetchBlob("xent"), equal_nan=False))


def test_maxpool():
    workspace.ResetWorkspace()

    # shape is in NCHW format
    # [[shape], kernel, stride] #TODO: add padding
    param_list = [[[1, 3, 10, 10], 2, 2],
                  [[2, 3, 5, 5], 1, 1],
                  [[2, 2, 7, 7], 3, 2],
                  [[8, 5, 8, 8], 4, 4]]

    for param_iter in param_list:
        shape, kernel, stride = param_iter
        data1 = [random.gauss(mu=0, sigma=10) for i in range(np.prod(shape))]

        net = core.Net("net")
        X = net.GivenTensorFill([], ["X"], shape=shape, values=data1, name="X")
        net.MaxPool(X, 'Y', kernel=kernel, stride=stride)

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


def test_avgpool():
    workspace.ResetWorkspace()

    # shape is in NCHW format
    # [[shape], kernel, stride] #TODO: add padding
    param_list = [[[1, 3, 10, 10], 2, 2],
                  [[2, 3, 5, 5], 1, 1],
                  [[2, 2, 7, 7], 3, 2],
                  [[8, 5, 8, 8], 4, 4]]

    for param_iter in param_list:
        shape, kernel, stride = param_iter
        data1 = [random.gauss(mu=0, sigma=10) for i in range(np.prod(shape))]

        net = core.Net("net")
        X = net.GivenTensorFill([], ["X"], shape=shape, values=data1, name="X")
        net.AveragePool(X, 'Y', kernel=kernel, stride=stride)

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
        assert(np.allclose(f_result, workspace.FetchBlob("Y"),
                           atol=1e-4, rtol=1e-3, equal_nan=False))
