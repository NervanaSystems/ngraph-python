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
from builtins import filter

import tensorflow as tf
import numpy as np
from ngraph.frontends.tensorflow.tests.importer_tester import ImporterTester
from ngraph.frontends.tensorflow.tests.test_util import TempDir
from ngraph.frontends.tensorflow.tf_importer.ops_nn import common_conv2d_pool_padding
import pytest
import itertools


def gen_conv_testcase():
    """
    Generate convolution test cases

    Returns:
        Iterator of (C, D, H, W, N, T, R, S, K, strides, padding)
    """

    def args_filter(args):
        """
        Filter currently not supported case (only symmetric padding allowed).

        Returns:
            True if args shall be kept.
        """
        C, D, H, W, N, T, R, S, K, strides, padding = args
        pad_t, pad_b, pad_l, pad_r = common_conv2d_pool_padding((N, H, W, C),
                                                                (R, S, C, K),
                                                                strides, padding)
        return pad_t == pad_b and pad_l == pad_r

    # test params
    Cs = [2, ]
    Ds = [1, ]
    Hs = [28, 11]
    Ws = [28, 11]
    Ns = [8, ]
    Ts = [1, ]
    Rs = [1, 3]
    Ss = [1, 3]
    Ks = [4, ]
    strides_list = [[1, 1, 1, 1], [1, 2, 3, 1]]
    paddings = ['SAME', 'VALID']
    all_args = list(
        itertools.product(Cs, Ds, Hs, Ws, Ns, Ts, Rs, Ss, Ks, strides_list,
                          paddings))
    return filter(args_filter, all_args)


def gen_pool_testcase():
    """
    Generate pooling test cases

    Returns:
        Iterator of (C, D, H, W, N, J, T, R, S, strides, padding)
    """

    def args_filter(args):
        """
        Filter currently not supported case (only symmetric padding allowed).

        Returns:
            True if args shall be kept.
        """
        C, D, H, W, N, J, T, R, S, strides, padding = args
        pad_t, pad_b, pad_l, pad_r = common_conv2d_pool_padding((N, H, W, C),
                                                                (R, S, C, C),
                                                                strides, padding)
        return pad_t == pad_b and pad_l == pad_r

    # test params
    Cs = [2, ]
    Ds = [1, ]
    Hs = [28, 11]
    Ws = [28, 11]
    Ns = [8, ]
    Js = [1, ]
    Ts = [1, ]
    Rs = [1, 3]
    Ss = [1, 3]
    strides_list = [[1, 1, 1, 1], [1, 2, 3, 1]]
    paddings = ['SAME', 'VALID']
    all_args = list(
        itertools.product(Cs, Ds, Hs, Ws, Ns, Js, Ts, Rs, Ss, strides_list,
                          paddings))
    return filter(args_filter, all_args)


class Tester(ImporterTester):

    @pytest.mark.parametrize("all_args", gen_conv_testcase())
    def test_conv(self, all_args):
        C, D, H, W, N, T, R, S, K, strides, padding = all_args
        image = tf.constant(np.random.rand(N, H, W, C).astype(np.float32))
        weight = tf.constant(np.random.rand(R, S, C, K).astype(np.float32))
        result = tf.nn.conv2d(image, weight, strides=strides, padding=padding)
        self.run(result, tf_feed_dict={}, rtol=1e-0, atol=1e-4)

    @pytest.mark.parametrize("all_args", gen_pool_testcase())
    def test_max_pooling(self, all_args):
        C, D, H, W, N, J, T, R, S, strides, padding = all_args
        ksize = (1, R, S, J)
        image = tf.constant(np.random.rand(N, H, W, C).astype(np.float32))
        result = tf.nn.max_pool(image, ksize, strides=strides, padding=padding)
        self.run(result, tf_feed_dict={}, rtol=1e-0, atol=1e-4)
    #
    # @pytest.mark.parametrize("all_args", gen_conv_testcase())
    # def test_bias_add(self, all_args):
    #     C, D, H, W, N, _, _, _, _, strides, padding = all_args
    #     image = tf.constant(np.random.rand(N, H, W, C).astype(np.float32))
    #     bias = tf.constant(np.random.rand(C).astype(np.float32))
    #     result = image + bias
    #     self.run(result, tf_feed_dict={}, rtol=1e-0, atol=1e-4)
    #
    # def test_sparse_softmax_cross_entropy_with_logits(self):
    #     # numpy random values
    #     np_logits = np.random.randn(128, 10).astype(np.float32)
    #     np_labels = np.random.randint(10, size=(128,))
    #
    #     # tf placeholders
    #     tf_logits = tf.placeholder(tf.float32, shape=np_logits.shape)
    #     tf_labels = tf.placeholder(tf.int32, shape=np_labels.shape)
    #
    #     # tf op
    #     tf_result_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #         tf_logits, tf_labels)
    #
    #     # feed_dict
    #     feed_dict = {tf_logits: np_logits, tf_labels: np_labels}
    #
    #     # test
    #     self.run(tf_result_op, tf_feed_dict=feed_dict)
    #
    # def test_softmax(self):
    #     # tf ops
    #     y = tf.placeholder(tf.float32, [8, 5])
    #     f = tf.nn.softmax(y)
    #     y_np = np.random.randn(8, 5)
    #     feed_dict = {y: y_np}
    #
    #     # test
    #     self.run(f, tf_feed_dict=feed_dict)
    #
    # def test_mnist_softmax_forward(self):
    #     # tf placeholder
    #     from tensorflow.examples.tutorials.mnist import input_data
    #     with TempDir() as tmpdir:
    #         mnist = input_data.read_data_sets(tmpdir, one_hot=True)
    #         x = tf.placeholder(tf.float32, [128, 784])
    #         W = tf.Variable(tf.zeros([784, 10]))
    #         b = tf.Variable(tf.zeros([10]))
    #         y = tf.matmul(x, W) + b
    #         y_ = tf.placeholder(tf.float32, [128, 10])
    #         cross_entropy = tf.reduce_mean(-tf.reduce_sum(
    #             y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
    #         init_op = tf.global_variables_initializer()
    #         batch_xs, batch_ys = mnist.train.next_batch(128)
    #
    #         # test
    #         feed_dict = {x: batch_xs, y_: batch_ys}
    #
    #         self.run(cross_entropy, tf_init_op=init_op, tf_feed_dict=feed_dict)
