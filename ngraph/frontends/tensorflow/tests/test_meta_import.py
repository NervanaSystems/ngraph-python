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
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.tensorflow.tf_importer.utils import SGDOptimizer

import pytest
from ngraph.frontends.tensorflow.tests.importer_tester import ImporterTester


class Args:
    """
    Default arguments
    """

    def __init__(self):
        self.data_dir = '/tmp/data'
        self.max_iter = 10
        self.lrate = 0.1
        self.batch_size = 128
        self.checkpoint_path = 'model.ckpt'


@pytest.mark.usefixtures("transformer_factory")
class TestMetaGraphWeightsImport(ImporterTester):
    @classmethod
    def setup_class(self):
        self.args = Args()  # default arguments

    @classmethod
    def teardown_method(self, method, delete_dump=True):
        # clear sess.graph_def
        tf.reset_default_graph()

        # remove dumped protobuf
        if delete_dump:
            # e.g. dir/checkpoint
            try:
                dir_name = os.path.dirname(
                    os.path.abspath(self.args.checkpoint_path))
                checkpoint_file_path = os.path.join(dir_name, "checkpoint")
                os.remove(checkpoint_file_path)
            except:
                print("[clean up] checkpoint does not exist")
            # e.g. dir/model.ckpt
            try:
                os.remove(self.args.checkpoint_path)
            except:
                print("[clean up] checkpoint (weights) does not exist")
            # e.g. dir/model.ckpt.meta
            try:
                os.remove(self.args.checkpoint_path + '.meta')
            except:
                print("[clean up] metagraph dump does not exist")

    def test_mnist_mlp_save_Load(self):
        # train
        self.train_mnist(self.args)
        # retrain
        ng_costs = self.ng_retrain_mnist(self.args)
        tf_costs = self.tf_retrain_mnist(self.args)
        # check results
        assert np.allclose(
            np.asarray(tf_costs).astype(np.float32),
            np.asarray(ng_costs).astype(np.float32))

    def train_mnist(self, args):
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
            train_op = tf.train.GradientDescentOptimizer(args.lrate).minimize(
                cost)
            init_op = tf.initialize_all_variables()

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

                # save
                save_path = saver.save(sess, args.checkpoint_path)
                print("Model saved in file: %s" % save_path)

    def ng_retrain_mnist(self, args):
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
        importer = ng.make_tf_importer()

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
        updates = SGDOptimizer(args.lrate).minimize(cost_ng)
        train_comp = transformer.computation([cost_ng, updates], x_ng, t_ng)
        init_comp = transformer.computation(init_op_ng)
        restore_comp = transformer.computation(restore_op_ng)
        transformer.initialize()

        # train in ngraph
        init_comp()
        restore_comp()
        costs = []
        for idx in range(args.max_iter):
            batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
            cost_val, _ = train_comp(batch_xs, batch_ys)
            print("[Iter %s] Cost = %s" % (idx, cost_val))
            costs.append(float(cost_val))
        return costs

    def tf_retrain_mnist(self, args):
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
            saver.restore(sess, args.checkpoint_path)

            # get op handle
            x = tf.get_collection('x')[0]
            t = tf.get_collection('t')[0]
            train_op = tf.get_collection('train_op')[0]
            cost = tf.get_collection('cost')[0]

            # train
            costs = []
            for idx in range(args.max_iter):
                batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
                cost_val, _ = sess.run([cost, train_op],
                                       feed_dict={x: batch_xs, t: batch_ys})
                print("[Iter %s] Cost = %s" % (idx, cost_val))
                costs.append(float(cost_val))
        return costs
