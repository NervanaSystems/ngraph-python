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

import os
import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import ngraph.transformers as ngt
from ngraph.frontends.tensorflow.tf_importer.importer import TFImporter
import ngraph.frontends.common.utils as util


def train_mnist(args):
    """
    Train in TF for max_iter, and save meta_graph / checkpoint

    Args:
        args: command line arguments
    """
    # dataset
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

    graph = tf.Graph()
    with graph.as_default():
        # write tensorflow models
        x = tf.placeholder(tf.float32, [args.batch_size, 784])
        t = tf.placeholder(tf.float32, [args.batch_size, 10])
        w = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.matmul(x, w) + b
        cost = tf.reduce_mean(-tf.reduce_sum(
            t * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
        train_op = tf.train.GradientDescentOptimizer(args.lrate).minimize(cost)
        init_op = tf.global_variables_initializer()

        # saver
        saver = tf.train.Saver()
        tf.add_to_collection('x', x)
        tf.add_to_collection('t', t)
        tf.add_to_collection('init_op', train_op)
        tf.add_to_collection('train_op', train_op)
        tf.add_to_collection('cost', cost)

        # train and save model
        with tf.Session() as sess:
            # init
            sess.run(init_op)

            # train
            for idx in range(args.max_iter):
                batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
                cost_val, _ = sess.run([cost, train_op],
                                       feed_dict={x: batch_xs, t: batch_ys})
                print("[Iter %s] Cost = %s" % (idx, cost_val))

            # save
            save_path = saver.save(sess, args.checkpoint_path)
            print("Model saved in file: %s" % save_path)


def ng_retrain_mnist(args):
    """
    Load meta_graph / checkpoint and retrain in ng for max_iter

    Args:
        args: command line arguments
    """

    # dataset, with offset max_iter
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
    for _ in range(args.max_iter):
        mnist.train.next_batch(args.batch_size)

    # init importer
    importer = TFImporter()

    # parse meta-graph and model checkpoint file
    importer.import_meta_graph(args.checkpoint_path + '.meta',
                               checkpoint_path=args.checkpoint_path)

    # get collections, must be specified by `tf.add_to_collection` before save
    x_ng, t_ng, cost_ng, init_op_ng = importer.get_collection_handle(
        ['x', 't', 'cost', 'init_op'])

    # get variable restore op
    restore_op_ng = importer.get_restore_op()

    # transformer and computations
    transformer = ngt.make_transformer()
    updates = util.CommonSGDOptimizer(args.lrate).minimize(cost_ng, cost_ng.variables())
    train_comp = transformer.computation([cost_ng, updates], x_ng, t_ng)
    init_comp = transformer.computation(init_op_ng)
    restore_comp = transformer.computation(restore_op_ng)
    transformer.initialize()

    # train in ngraph
    init_comp()
    restore_comp()
    ng_cost_vals = []
    for idx in range(args.max_iter):
        batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
        cost_val, _ = train_comp(batch_xs, batch_ys)
        ng_cost_vals.append(float(cost_val))
        print("[Iter %s] Cost = %s" % (idx, cost_val))

    transformer.close()

    return ng_cost_vals


def tf_retrain_mnist(args):
    """
    Load meta_graph / checkpoint and retrain in TF for max_iter (comparision)

    Args:
        args: command line arguments
    """

    # dataset, with offset max_iter
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
    for _ in range(args.max_iter):
        mnist.train.next_batch(args.batch_size)

    # saver
    saver = tf.train.import_meta_graph(args.checkpoint_path + '.meta')

    # load model
    with tf.Session() as sess:
        # restore model
        saver.restore(sess, os.path.join(os.getcwd(), 'model.ckpt'))

        # get op handle
        x = tf.get_collection('x')[0]
        t = tf.get_collection('t')[0]
        train_op = tf.get_collection('train_op')[0]
        cost = tf.get_collection('cost')[0]

        # train
        tf_cost_vals = []
        for idx in range(args.max_iter):
            batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
            cost_val, _ = sess.run([cost, train_op],
                                   feed_dict={x: batch_xs, t: batch_ys})
            tf_cost_vals.append(float(cost_val))
            print("[Iter %s] Cost = %s" % (idx, cost_val))

    return tf_cost_vals


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='/tmp/data')
    parser.add_argument('-i', '--max_iter', type=int, default=10)
    parser.add_argument('-l', '--lrate', type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-s', '--checkpoint_path', default='model.ckpt')
    args = parser.parse_args()

    # train in TF for max_iter, and save meta_graph / checkpoint
    train_mnist(args)

    # load meta_graph / checkpoint and retrain in ng for max_iter
    ng_retrain_mnist(args)

    # load meta_graph / checkpoint and retrain in TF for max_iter (verification)
    tf_retrain_mnist(args)
