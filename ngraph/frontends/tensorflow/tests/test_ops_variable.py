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

import tensorflow as tf
import numpy as np
import ngraph as ng
from ngraph.frontends.tensorflow.tests.importer_tester import ImporterTester
import pytest

pytestmark = pytest.mark.transformer_dependent


class Tester(ImporterTester):
    def test_variable(self):
        # tf placeholder
        a = tf.Variable(tf.constant(np.random.randn(2, 3), name="a"))
        b = tf.Variable(tf.constant(np.random.randn(2, 3), name="b"))
        init_op = tf.global_variables_initializer()
        result = tf.add(a, b) * 3

        # test
        self.run(result, tf_init_op=init_op)

    @pytest.mark.xfail(strict=True)
    def test_ref_assign(self):
        # Currently ngraph and tf have different assign semantics
        # eval(ng.assign(a, 1)) resturns None, but eval(tf.assign(a, 1)) returns
        # a which is 1.
        # TODO: fix this test after assign op / user_deps are fixed in ngraph
        # TODO: double assignments fails

        # tf placeholder
        a = tf.Variable(tf.constant(np.random.randn(2, 3), name="a"))
        b = tf.Variable(tf.constant(np.random.randn(2, 3), name="b"))
        init_op = tf.global_variables_initializer()
        a_update = tf.assign(a, b)

        # test
        tf_result = self.tf_run(a_update, tf_init_op=init_op)
        ng_result = self.ng_run(a)
        ng.testing.assert_allclose(tf_result, ng_result)

    @pytest.mark.xfail(strict=True)
    def test_ref_assign_add(self):
        # Currently ngraph and tf have different assign semantics
        # eval(ng.assign(a, 1)) resturns None, but eval(tf.assign(a, 1)) returns
        # a which is 1.
        # TODO: fix this test after assign op / user_deps are fixed in ngraph
        # TODO: double assignments fails

        # tf placeholder
        a = tf.Variable(tf.constant(np.random.randn(2, 3), name="a"))
        b = tf.Variable(tf.constant(np.random.randn(2, 3), name="b"))
        init_op = tf.global_variables_initializer()
        a_update = tf.assign_add(a, b)

        # test
        tf_result = self.tf_run(a_update, tf_init_op=init_op)
        ng_result = self.ng_run(a)
        ng.testing.assert_allclose(tf_result, ng_result)
