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


def test_maxpool():
    workspace.ResetWorkspace()

    shape = [2, 3, 10, 10]  # NCHW
    data1 = [random.gauss(mu=0, sigma=10) for i in range(np.prod(shape))]

    net = core.Net("net")
    X = net.GivenTensorFill([], ["X"], shape=shape, values=data1, name="X")
    net.MaxPool(X, 'Y', kernel=2, stride=2)

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

    shape = [2, 3, 10, 10]  # NCHW
    data1 = [random.gauss(mu=0, sigma=10) for i in range(np.prod(shape))]

    net = core.Net("net")
    X = net.GivenTensorFill([], ["X"], shape=shape, values=data1, name="X")
    net.AveragePool(X, 'Y', kernel=2, stride=2)

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


def test_convolution_nhwc_no_pad_no_bias():
    workspace.ResetWorkspace()

    # shape is in NCHW format
    # [batch, input_feature_map, spatial, output_feature_map, kernel, stride]
    param_list = [
                  [1, 3, 2, 1, 2, 2],
                  [1, 1, 4, 1, 2, 2],
                  [2, 3, 8, 1, 2, 2],
                  [8, 2, 5, 4, 3, 1],
                 ]


    for param_iter in param_list:
        n, ifm, spatial, ofm, kernel, stride = param_iter

        shape_x = (n, spatial, spatial, ifm)
        shape_w = (ofm, kernel, kernel, ifm)
        shape_b = (ofm, )

        data_x = [random.gauss(mu=0, sigma=10) for i in range(np.prod(shape_x))]
        data_w = [random.gauss(mu=0, sigma=10) for i in range(np.prod(shape_w))]
        data_b = [0. for i in range(np.prod(shape_b))]

        net = core.Net("net")
        X = net.GivenTensorFill([], ["X"], shape=shape_x, values=data_x, name="X")
        W = net.GivenTensorFill([], ["W"], shape=shape_w, values=data_w, name="W")
        B = net.GivenTensorFill([], ["B"], shape=shape_b, values=data_b, name="B")

        net.Conv([X, W, B], 'Y', kernel=kernel, stride=stride, order='NHWC')

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


def test_convolution_nchw_no_pad_no_bias():
    # [batch, input_feature_map, spatial, output_feature_map, kernel, stride]
    param_list = [
        [1, 3, 2, 1, 2, 2],
        [1, 1, 4, 1, 2, 2],
        [2, 3, 8, 1, 2, 2],
        [8, 2, 5, 4, 3, 1],
    ]

    for param_iter in param_list:
        n, ifm, spatial, ofm, kernel, stride = param_iter

        shape_x = (n, ifm, spatial, spatial)
        shape_w = (ofm, ifm, kernel, kernel)
        shape_b = (ofm,)

        data_x = [random.gauss(mu=0, sigma=10) for i in range(np.prod(shape_x))]
        data_w = [random.gauss(mu=0, sigma=10) for i in range(np.prod(shape_w))]
        data_b = [0. for i in range(np.prod(shape_b))]

        net = core.Net("net")
        X = net.GivenTensorFill([], ["X"], shape=shape_x, values=data_x, name="X")
        W = net.GivenTensorFill([], ["W"], shape=shape_w, values=data_w, name="W")
        B = net.GivenTensorFill([], ["B"], shape=shape_b, values=data_b, name="B")

        net.Conv([X, W, B], 'Y', kernel=kernel, stride=stride, order='NCHW')

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
        assert (np.allclose(f_result, workspace.FetchBlob("Y"), atol=1e-4, rtol=1e-3, equal_nan=False))
