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


def run_all_close_compare_initiated_with_random_gauss(c2_op_name,
                                                      shape=None,
                                                      data=None,
                                                      expected=None):
    workspace.ResetWorkspace()
    if not shape:
        shape = [2, 7]
    if not data:
        data = [random.gauss(mu=0, sigma=10) for i in range(np.prod(shape))]

    net = core.Net("net")
    net.GivenTensorFill([], "X", shape=shape, values=data, name="X")
    getattr(net, c2_op_name)(["X"], ["Y"], name="Y")

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

    c2_y = workspace.FetchBlob("Y")

    # compare Caffe2 and ngraph results
    assert(np.allclose(f_result, c2_y, atol=1e-4, rtol=0, equal_nan=False))

    # compare expected results and ngraph results
    if expected:
        assert(np.allclose(f_result, expected, atol=1e-3, rtol=0, equal_nan=False))


def test_relu():
    run_all_close_compare_initiated_with_random_gauss('Relu',
                                                      shape=[10, 10])


def test_softmax():
    shape = [2, 7]
    data = [
        1., 2., 3., 4., 1., 2., 3.,
        1., 2., 3., 4., 1., 2., 3.
    ]
    expected = [
        [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175],
        [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175],
    ]
    run_all_close_compare_initiated_with_random_gauss('Softmax',
                                                      shape=shape,
                                                      data=data,
                                                      expected=expected)


def test_negative():
    run_all_close_compare_initiated_with_random_gauss('Negative')


def test_sigmoid():
    run_all_close_compare_initiated_with_random_gauss('Sigmoid')


def test_tanh():
    run_all_close_compare_initiated_with_random_gauss('Tanh')
