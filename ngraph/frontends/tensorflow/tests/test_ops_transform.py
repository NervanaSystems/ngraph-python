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
from ngraph.frontends.tensorflow.tf_importer.utils import tf_obj_shape, \
    get_nested_attr
import pytest

pytestmark = pytest.mark.transformer_dependent


class Tester(ImporterTester):
    @pytest.mark.parametrize("shape", [(1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
    @pytest.mark.parametrize("op_name", ["rank", "shape", "size"])
    def test_rank_size_shape(self, shape, op_name):
        # tf placeholders
        x = tf.placeholder(tf.float32, shape=shape)

        # tf op
        tf_op = get_nested_attr(tf, op_name)
        f = tf_op(x)

        # values
        feed_dict = dict()
        feed_dict[x] = np.random.rand(*tf_obj_shape(x))

        # test
        self.run(f, tf_feed_dict=feed_dict)

    def test_range(self):
        # shapes to test
        test_params = [(1, 2, 10), (3, 5, 18)]

        # test
        for params in test_params:
            f = tf.range(*params)
            self.run(f)

    @pytest.mark.parametrize("shape", [(360,), (3, 120), (12, 30), (60, 6)])
    def test_reshape(self, shape):
        # TODO: currently generic reshape is not supported in ngraph yet

        # const reshape
        x = tf.constant(
            np.random.randn(3, 4, 5, 6).astype(np.float32), dtype=tf.float32)
        x_reshaped = tf.reshape(x, shape)
        self.run(x_reshaped, tf_feed_dict={})

    @pytest.mark.parametrize("shape", [(360,), (3, 120), (12, 30), (60, 6)])
    def test_reshape_flatten(self, shape):
        # TODO: only support 1d to 2d reshape for non constant by flattening
        # flatten to 1d or 2d
        x = tf.Variable(
            np.random.randn(3, 4, 5, 6).astype(np.float32), dtype=tf.float32)
        init_op = tf.global_variables_initializer()
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
