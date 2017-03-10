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
import pytest


@pytest.mark.transformer_dependent
class Tester(ImporterTester):
    def test_rank(self):
        # shapes to test
        shapes = [(1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)]

        # tf placeholders
        placeholders = [tf.placeholder(tf.float32, shape=s) for s in shapes]

        # ranks
        ranks = [tf.rank(p) for p in placeholders]

        # values
        feed_dict = dict()
        for x in placeholders:
            feed_dict[x] = np.random.rand(*tf_to_shape_tuple(x))

        # test
        for rank in ranks:
            self.run(rank, tf_feed_dict=feed_dict)

    def test_range(self):
        # shapes to test
        test_params = [(1, 2, 10), (3, 5, 18)]

        # test
        for params in test_params:
            f = tf.range(*params)
            self.run(f)

    def test_size(self):
        # shapes to test
        shapes = [(1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)]

        # tf placeholders
        placeholders = [tf.placeholder(tf.float32, shape=s) for s in shapes]

        # ranks
        sizes = [tf.size(p) for p in placeholders]

        # values
        feed_dict = dict()
        for x in placeholders:
            feed_dict[x] = np.random.rand(*tf_to_shape_tuple(x))

        # test
        for size in sizes:
            self.run(size, tf_feed_dict=feed_dict)

    def test_shape(self):
        # shapes to test
        shapes = [(1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)]

        # tf placeholders
        placeholders = [tf.placeholder(tf.float32, shape=s) for s in shapes]

        # ranks
        result_ops = [tf.shape(p) for p in placeholders]

        # values
        feed_dict = dict()
        for x in placeholders:
            feed_dict[x] = np.random.rand(*tf_to_shape_tuple(x))

        # test
        for op in result_ops:
            self.run(op, tf_feed_dict=feed_dict)

    def test_reshape(self):
        # TODO: currently generic reshape is not supported in ngraph yet
        # shapes
        shapes = [(360,), (3, 120), (12, 30), (60, 6)]

        # const reshape
        x = tf.constant(
            np.random.randn(3, 4, 5, 6).astype(np.float32), dtype=tf.float32)
        for shape in shapes:
            x_reshaped = tf.reshape(x, shape)
            self.run(x_reshaped, tf_feed_dict={})

    def test_reshape_flatten(self):
        # TODO: currently only supports flatten to 1d or 2d
        shapes = [(360,), (3, 120), (12, 30), (60, 6)]

        # flatten to 1d or 2d
        x = tf.Variable(
            np.random.randn(3, 4, 5, 6).astype(np.float32), dtype=tf.float32)
        init_op = tf.global_variables_initializer()
        for shape in shapes:
            x_reshaped = tf.reshape(x, shape)
            self.run(x_reshaped, tf_init_op=init_op, tf_feed_dict={})

    def test_tile(self):
        # TODO: currently Tile not supported in ngrpah, only test const case
        x = tf.constant(
            np.random.randn(2, 3).astype(np.float32), dtype=tf.float32)
        multiples = tf.constant([2, 4], dtype=tf.int32)

        # test
        self.run(tf.tile(x, multiples))

    def test_expand_dims(self):
        x = tf.constant(
            np.random.randn(2, ).astype(np.float32), dtype=tf.float32)
        dims = [-2, -1, 0, 1]
        for dim in dims:
            self.run(tf.expand_dims(x, dim))

        x = tf.constant(
            np.random.randn(2, 3, 5).astype(np.float32), dtype=tf.float32)
        dims = [-4, -3, -2, -1, 0, 1, 2, 3]
        for dim in dims:
            self.run(tf.expand_dims(x, dim))
