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
from tensorflow.examples.tutorials.mnist import input_data
from ngraph.frontends.tensorflow.tf_importer.utils import SGDOptimizer
from ngraph.frontends.tensorflow.tf_importer.importer import TFImporter
import ngraph.transformers as ngt

import pytest


@pytest.mark.usefixtures("transformer_factory")
class TestMnist(ImporterTester):
    def test_mnist_softmax(self):
        # parameters
        max_iter = 10
        lrate = 0.1
        bsz = 128

        # write tensorflow models
        x = tf.placeholder(tf.float32, [bsz, 784])
        t = tf.placeholder(tf.float32, [bsz, 10])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.matmul(x, W) + b
        cost = tf.reduce_mean(-tf.reduce_sum(
            t * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
        init_op = tf.initialize_all_variables()

        # import graph_def
        importer = TFImporter()
        importer.import_graph_def(tf.get_default_graph().as_graph_def())

        # get handle of ngraph ops
        x_ng, t_ng, cost_ng, init_op_ng = importer.get_op_handle(
            [x, t, cost, init_op])

        # transformer and computations
        transformer = ngt.make_transformer()
        updates = SGDOptimizer(lrate).minimize(cost_ng)
        train_comp = transformer.computation([cost_ng, updates], x_ng, t_ng)
        init_comp = transformer.computation(init_op_ng)
        transformer.initialize()

        # train
        mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
        init_comp()
        ng_costs = []
        for idx in range(max_iter):
            batch_xs, batch_ys = mnist.train.next_batch(bsz)
            cost_val, _ = train_comp(batch_xs, batch_ys)
            print("[Iter %s] Cost = %s" % (idx, cost_val))
            ng_costs.append(float(cost_val))

        # train in tensorflow as comparison
        with tf.Session() as sess:
            # train in tensorflow
            train_step = tf.train.GradientDescentOptimizer(lrate).minimize(
                cost)
            sess.run(init_op)

            mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
            tf_costs = []
            for idx in range(max_iter):
                batch_xs, batch_ys = mnist.train.next_batch(bsz)
                cost_val, _ = sess.run([cost, train_step],
                                       feed_dict={x: batch_xs,
                                                  t: batch_ys})
                print("[Iter %s] Cost = %s" % (idx, cost_val))
                tf_costs.append(cost_val)

        # check results
        assert np.allclose(
            np.asarray(tf_costs).astype(np.float32),
            np.asarray(ng_costs).astype(np.float32))
