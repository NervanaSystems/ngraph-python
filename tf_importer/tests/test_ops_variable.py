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
from tf_importer.tests.importer_tester import ImporterTester


class Tester(ImporterTester):

    def test_variable(self):
        # tf placeholder
        a = tf.Variable(tf.ones([2, 3]), name="a")
        b = tf.Variable(tf.zeros([1, 3]), name="b")
        init_op = tf.initialize_all_variables()
        result = tf.add(a, b) * 3

        # test
        self.run(result, tf_init_op=init_op)

    def test_assign(self):
        # TODO: double assignments fails

        # tf placeholder
        a = tf.Variable(tf.ones([2, 3]) * 3, name="a")
        b = tf.Variable(tf.zeros([2, 3]) + tf.ones([2, 3]) * 2, name="b")
        init_op = tf.initialize_all_variables()
        a_update = tf.assign(a, b)

        # test
        tf_result = self.tf_run(a_update, tf_init_op=init_op)
        ng_result = self.ng_run(a)
        assert np.allclose(tf_result, ng_result)

    def test_assign_add(self):
        # TODO: double assignments fails

        # tf placeholder
        a = tf.Variable(tf.ones([2, 3]) * 3, name="a")
        b = tf.Variable(tf.zeros([2, 3]) + tf.ones([2, 3]) * 2, name="b")
        init_op = tf.initialize_all_variables()
        a_update = tf.assign_add(a, b)

        # test
        tf_result = self.tf_run(a_update, tf_init_op=init_op)
        ng_result = self.ng_run(a)
        assert np.allclose(tf_result, ng_result)
