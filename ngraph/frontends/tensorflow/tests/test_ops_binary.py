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


@pytest.mark.transformer_dependent
class Tester(ImporterTester):
    @pytest.mark.parametrize("op_name", [
        'add',
        'sub',
        'mul',
        'div',
        'maximum'
    ])
    def test_binary_ops(self, op_name):
        a = tf.placeholder(tf.float32, shape=(2, 3))
        b = tf.placeholder(tf.float32, shape=(2, 3))
        tf_op = get_nested_attr(tf, op_name)
        f = tf_op(a, b)

        # value
        a_val = np.random.rand(*tf_obj_shape(a))
        b_val = np.random.rand(*tf_obj_shape(b))

        # test
        self.run(f, tf_feed_dict={a: a_val, b: b_val})

    @pytest.mark.parametrize("shapes", [
        [(2, 1), (1,)],
        [(3, 2), (2,)],
        [(3, 2, 1), (1,)],
        [(4, 3, 2), (2,)]
    ])
    def test_bias_add(self, shapes):
        a_shape, b_shape = shapes
        a = tf.placeholder(tf.float32, shape=a_shape)
        b = tf.placeholder(tf.float32, shape=b_shape)
        f = tf.nn.bias_add(a, b)

        # value
        a_val = np.random.rand(*tf_obj_shape(a))
        b_val = np.random.rand(*tf_obj_shape(b))

        # test
        self.run(f, tf_feed_dict={a: a_val, b: b_val})

    def test_mod(self):
        # computation
        a = tf.placeholder(tf.int32, shape=(6,))
        b = tf.placeholder(tf.int32, shape=(6,))
        f = a % b

        # value
        a_val = np.array([0, 10, 11, 12, 13, 14], dtype=np.int32)
        b_val = np.array([5, 5, 5, 5, 5, 5], dtype=np.int32)

        # test
        self.run(f, tf_feed_dict={a: a_val, b: b_val})
