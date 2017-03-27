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


from ngraph.frontends.caffe2.examples.linear_regression import linear_regression
from ngraph.frontends.caffe2.examples.fc import fc_example
from ngraph.frontends.caffe2.examples.mnist_mlp import mnist_mlp
from ngraph.frontends.caffe2.examples.sum import sum_example
import pytest


# Note that these tests are only unit tests. Check if workload is consistent and can be run.
# These tests don't check results.


def test_linear_regression():
    iter_num, lrate, gamma, step_size, noise_scale = 200, 0.01, 0.9, 20, 0.01
    linear_regression(iter_num, lrate, gamma, step_size, noise_scale)


def test_fc_example():
    fc_example()


def test_sum_example():
    sum_example()


# We don't want to attempt to run any tests that attempt to download data or write temporary
# files as part of the merge-blocking unit test suites
@pytest.mark.skip()
def test_mnist_mlp():
    args = type('', (), {})()
    args.data_dir = '/tmp/data'
    args.max_iter = 2
    args.lrate = 0.01
    args.batch = 16
    args.verbose = 0
    # fixed inv policy
    args.power = 0.75
    args.gamma = 0.0001

    mnist_mlp(args)
