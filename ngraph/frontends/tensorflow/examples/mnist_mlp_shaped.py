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

from __future__ import division
from __future__ import print_function

from ngraph.testing import ExecutorFactory
from tensorflow.examples.tutorials.mnist import input_data
from ngraph.frontends.common.utils import CommonSGDOptimizer
import numpy as np
import tensorflow as tf
import ngraph as ng
import ngraph.frontends.tensorflow.tf_importer.ngraph_shaped as ns
import argparse


def mnist_mlp_ns(args):
    # write tensorflow models
    x = ns.placeholder(np.float32, [args.batch_size, 784])
    t = ns.placeholder(np.float32, [args.batch_size, 10])
    w = ns.Variable(np.zeros([784, 10]))
    b = ns.Variable(np.zeros([10]))
    y = ns.add(ns.matmul(x, w), b)
    cost = ns.reduce_mean(-ns.reduce_sum(ns.multiply(t, ns.log(ns.softmax(y))),
                                         axis=[1]))
    # transformer and computations
    with ExecutorFactory() as ex:
        updates = CommonSGDOptimizer(args.lrate).minimize(cost,
                                                          cost.variables())
        train_comp = ex.executor(ng.sequential([updates, cost]), x, t)
        ex.transformer.initialize()

        # train
        if args.random_data is not None:
            mnist = args.random_data
            mnist.reset(0)
        else:
            mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

        ng_cost_vals = []
        for idx in range(args.max_iter):
            batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
            cost_val = train_comp(batch_xs, batch_ys)
            ng_cost_vals.append(float(cost_val))
            print("[Iter %s] Cost = %s" % (idx, cost_val))

    return ng_cost_vals


def mnist_mlp_tf(args):
    # write tensorflow models
    x = tf.placeholder(tf.float32, [args.batch_size, 784])
    t = tf.placeholder(tf.float32, [args.batch_size, 10])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, w) + b
    cost = tf.reduce_mean(-tf.reduce_sum(tf.multiply(t, tf.log(tf.nn.softmax(y))),
                                         axis=[1]))
    init = tf.global_variables_initializer()

    # train in tensorflow as comparison
    with tf.Session() as sess:
        # train in tensorflow
        train_step = tf.train.GradientDescentOptimizer(args.lrate).minimize(
            cost)
        sess.run(init)
        if args.random_data is not None:
            mnist = args.random_data
            mnist.reset(0)
        else:
            mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
        tf_cost_vals = []
        for idx in range(args.max_iter):
            batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
            cost_val, _ = sess.run([cost, train_step],
                                   feed_dict={x: batch_xs, t: batch_ys})
            tf_cost_vals.append(float(cost_val))
            print("[Iter %s] Cost = %s" % (idx, cost_val))

    return tf_cost_vals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='/tmp/data')
    parser.add_argument('-i', '--max_iter', type=int, default=10)
    parser.add_argument('-l', '--lrate', type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('--random_data', default=None)
    args = parser.parse_args()
    mnist_mlp_ns(args)
    mnist_mlp_tf(args)
