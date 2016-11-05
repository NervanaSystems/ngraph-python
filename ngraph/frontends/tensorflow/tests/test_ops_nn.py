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


class Tester(ImporterTester):
    def test_sparse_softmax_cross_entropy_with_logits(self):
        # numpy random values
        np_logits = np.random.randn(128, 10).astype(np.float32)
        np_labels = np.random.randint(10, size=(128, ))

        # tf placeholders
        tf_logits = tf.placeholder(tf.float32, shape=np_logits.shape)
        tf_labels = tf.placeholder(tf.int32, shape=np_labels.shape)

        # tf op
        tf_result_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf_logits, tf_labels)

        # feed_dict
        feed_dict = {tf_logits: np_logits, tf_labels: np_labels}

        # test
        self.run(tf_result_op, tf_feed_dict=feed_dict)

    def test_softmax(self):
        # tf ops
        y = tf.placeholder(tf.float32, [8, 5])
        f = tf.nn.softmax(y)
        y_np = np.random.randn(8, 5)
        feed_dict = {y: y_np}

        # test
        self.run(f, tf_feed_dict=feed_dict)

    def test_mnist_softmax_forward(self):
        # tf placeholder
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
        x = tf.placeholder(tf.float32, [128, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.matmul(x, W) + b
        y_ = tf.placeholder(tf.float32, [128, 10])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(
            y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
        init_op = tf.initialize_all_variables()
        batch_xs, batch_ys = mnist.train.next_batch(128)

        # test
        feed_dict = {x: batch_xs, y_: batch_ys}

        self.run(cross_entropy, tf_init_op=init_op, tf_feed_dict=feed_dict)
