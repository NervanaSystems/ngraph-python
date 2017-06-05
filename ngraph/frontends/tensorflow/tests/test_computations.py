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

"""
Test combination of ops. Originally moved from other unit tests to this file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

from ngraph.frontends.tensorflow.tests.importer_tester import ImporterTester
from ngraph.frontends.tensorflow.tf_importer.utils import tf_obj_shape


@pytest.mark.transformer_dependent
class Tester(ImporterTester):
    def test_binary_ops_combined(self):
        # computation
        a = tf.placeholder(tf.float32, shape=(2, 3))
        b = tf.placeholder(tf.float32, shape=(2, 3))
        c = tf.add(a, b)
        d = tf.mul(c, a)
        e = tf.div(d, b)
        f = tf.sub(a, e)
        g = tf.maximum(a, f)

        # value
        a_val = np.random.rand(*tf_obj_shape(a))
        b_val = np.random.rand(*tf_obj_shape(b))

        # test
        self.run(g, tf_feed_dict={a: a_val, b: b_val})

    def test_broadcast_rules(self):
        # tf have un-implemented broadcasts
        # for example: (2, 1, 2, 1) + (1, 2, 1, 2) is not implemented in tf
        #              (10, 1, 2, 1, 5) + (11, 1, 1, 5) is not implemented in tf

        a = tf.placeholder(tf.float32, shape=(5, 1, 1, 4))
        b = tf.placeholder(tf.float32, shape=(1, 1, 3, 1, 1))
        c = tf.placeholder(tf.float32, shape=(1, 1, 4))
        d = tf.placeholder(tf.float32, shape=(4,))
        f = a + b * c + d

        # value
        feed_dict = dict()
        for x in [a, b, c, d]:
            feed_dict[x] = np.random.rand(*tf_obj_shape(x))

        # test
        self.run(f, tf_feed_dict=feed_dict)

    def test_constant_broadcast(self):
        # computation
        a = tf.constant([1, 2, 3.])
        b = tf.constant(20.)
        c = tf.ones([2, 3])
        d = tf.ones([10, 2, 3])
        e = tf.ones([1, 2, 3])
        f = a + b + c + d + e

        # test
        self.run(f)

    def test_sum_prod_broadcast(self):
        # placeholder
        a = tf.placeholder(tf.float32, shape=[3, 4, 5, 6])
        b = tf.placeholder(tf.float32, shape=[3, 4, 5])
        a_sum = tf.reduce_sum(a, reduction_indices=[0, 3])  # shape (4, 5)
        b_prod = tf.reduce_prod(b, reduction_indices=[0, 1])  # shape (5,)
        f = a_sum + b_prod + b  # (4, 5) + (5,) + (3, 4, 5) -> (3, 4, 5)

        # value
        feed_dict = dict()
        for x in [a, b]:
            feed_dict[x] = np.random.rand(*tf_obj_shape(x))

        # test
        self.run(f, tf_feed_dict=feed_dict)
