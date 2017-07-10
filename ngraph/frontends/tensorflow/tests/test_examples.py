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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ngraph as ng
from ngraph.frontends.tensorflow.tests.importer_tester import ImporterTester
from ngraph.frontends.tensorflow.tests.utils import FakeMNIST
import argparse

import pytest
from ngraph.frontends.tensorflow.examples.logistic_regression import \
    logistic_regression
from ngraph.frontends.tensorflow.examples.mnist_mlp import mnist_mlp
from ngraph.frontends.tensorflow.examples.mnist_mlp_shaped import \
    mnist_mlp_ns, mnist_mlp_tf

pytestmark = pytest.mark.transformer_dependent


@pytest.mark.usefixtures("transformer_factory")
class TestExamples(ImporterTester):
    # Failing test for Flex because of the strict tolerance (rtol, atol)
    @pytest.mark.flex_disabled
    def test_logistic_regression(self):
        # args
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--max_iter', type=int, default=10)
        parser.add_argument('-l', '--lrate', type=float, default=0.1,
                            help="Learning rate")
        args = parser.parse_args("")

        # compute
        ng_cost_vals, tf_cost_vals = logistic_regression(args)

        # check
        assert ng.testing.allclose(
            np.asarray(ng_cost_vals).astype(np.float32),
            np.asarray(tf_cost_vals).astype(np.float32))

    @pytest.mark.flex_disabled
    def test_mnist_mlp(self):
        # args
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--data_dir', default=None)
        parser.add_argument('-i', '--max_iter', type=int, default=10)
        parser.add_argument('-l', '--lrate', type=float, default=0.1,
                            help="Learning rate")
        parser.add_argument('-b', '--batch_size', type=int, default=128)
        parser.add_argument('--random_data', default=FakeMNIST())
        args = parser.parse_args("")

        # compute
        ng_cost_vals, tf_cost_vals = mnist_mlp(args)

        # check
        assert ng.testing.allclose(
            np.asarray(ng_cost_vals).astype(np.float32),
            np.asarray(tf_cost_vals).astype(np.float32))

    @pytest.mark.flex_disabled
    def test_mnist_mlp_shaped(self):
        # args
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--data_dir', default=None)
        parser.add_argument('-i', '--max_iter', type=int, default=10)
        parser.add_argument('-l', '--lrate', type=float, default=0.1,
                            help="Learning rate")
        parser.add_argument('-b', '--batch_size', type=int, default=128)
        parser.add_argument('--random_data', default=FakeMNIST())
        args = parser.parse_args("")

        # compute
        ns_cost_vals, tf_cost_vals = mnist_mlp_ns(args), mnist_mlp_tf(args)

        # check
        assert ng.testing.allclose(
            np.asarray(ns_cost_vals).astype(np.float32),
            np.asarray(tf_cost_vals).astype(np.float32))
