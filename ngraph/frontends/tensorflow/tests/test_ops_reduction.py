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
from ngraph.frontends.tensorflow.tests.importer_tester import ImporterTester
from ngraph.frontends.tensorflow.tf_importer.utils import tf_to_shape_tuple


class Tester(ImporterTester):
    def test_sum_prod_mean(self):
        # test cases
        reduction_indices_list = [None, [], [0, ], [0, 1], [1, 2], [0, 1, 2]]

        # tf placeholder
        a = tf.placeholder(tf.float32, shape=[3, 4, 5])

        # value
        feed_dict = {a: np.random.rand(*tf_to_shape_tuple(a))}

        # test
        for reduction_indices in reduction_indices_list:
            f = tf.reduce_sum(a, reduction_indices=reduction_indices)
            self.run(f, tf_feed_dict=feed_dict)
            f = tf.reduce_prod(a, reduction_indices=reduction_indices)
            self.run(f, tf_feed_dict=feed_dict)
            f = tf.reduce_mean(a, reduction_indices=reduction_indices)
            self.run(f, tf_feed_dict=feed_dict)

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
            feed_dict[x] = np.random.rand(*tf_to_shape_tuple(x))

        # test
        self.run(f, tf_feed_dict=feed_dict)
