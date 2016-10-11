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
from tf_importer.tf_importer.utils import tensor_shape_to_tuple


class Tester(ImporterTester):
    def test_tanh_sigmoid(self):
        # computation
        a = tf.placeholder(tf.float32, shape=(2, 3))
        b = tf.placeholder(tf.float32, shape=(3, 4))
        c = tf.placeholder(tf.float32, shape=(2, 4))
        d = tf.sigmoid(tf.matmul(a, tf.tanh(b)))
        e = tf.sigmoid(c)
        f = d + e

        # value
        feed_dict = dict()
        for x in [a, b, c]:
            feed_dict[x] = np.random.rand(*tensor_shape_to_tuple(x._shape))

        # test
        self.run(f, tf_feed_dict=feed_dict)

    def test_relu(self):
        # computation
        a = tf.placeholder(tf.float32, shape=(100, 200))
        f = tf.nn.relu(a)

        # value
        feed_dict = {a: np.random.randn(*tensor_shape_to_tuple(a._shape))}

        # test
        self.run(f, tf_feed_dict=feed_dict)

    def test_identity(self):
        # computation
        a = tf.placeholder(tf.float32, shape=(2, 3))
        b = tf.placeholder(tf.float32, shape=(2, 3))
        c = tf.identity(a) + b
        f = tf.identity(c)

        # value
        feed_dict = dict()
        for x in [a, b]:
            feed_dict[x] = np.random.rand(*tensor_shape_to_tuple(x._shape))

        # test
        self.run(f, tf_feed_dict=feed_dict)
