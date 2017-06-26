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
from ngraph.frontends.tensorflow.tf_importer.utils import tf_obj_shape
import pytest


@pytest.mark.transformer_dependent
class Tester(ImporterTester):
    def test_matmul(self):
        # computation
        a = tf.placeholder(tf.float32, shape=(2, 3))
        b = tf.placeholder(tf.float32, shape=(3, 4))
        c = tf.matmul(a, b)

        # value
        a_val = np.random.rand(*tf_obj_shape(a))
        b_val = np.random.rand(*tf_obj_shape(b))

        # test
        self.run(c, tf_feed_dict={a: a_val, b: b_val})

    def test_matmul_transpose(self):
        # case 1
        a = tf.placeholder(tf.float32, shape=(2, 3))
        b = tf.placeholder(tf.float32, shape=(3, 4))
        a_val = np.random.rand(*tf_obj_shape(a))
        b_val = np.random.rand(*tf_obj_shape(b))
        self.run(tf.matmul(a, b, transpose_a=False, transpose_b=False),
                 tf_feed_dict={a: a_val, b: b_val})

        # case 2
        a = tf.placeholder(tf.float32, shape=(3, 2))
        b = tf.placeholder(tf.float32, shape=(3, 4))
        a_val = np.random.rand(*tf_obj_shape(a))
        b_val = np.random.rand(*tf_obj_shape(b))
        self.run(tf.matmul(a, b, transpose_a=True, transpose_b=False),
                 tf_feed_dict={a: a_val, b: b_val})

        # case 3
        a = tf.placeholder(tf.float32, shape=(2, 3))
        b = tf.placeholder(tf.float32, shape=(4, 3))
        a_val = np.random.rand(*tf_obj_shape(a))
        b_val = np.random.rand(*tf_obj_shape(b))
        self.run(tf.matmul(a, b, transpose_a=False, transpose_b=True),
                 tf_feed_dict={a: a_val, b: b_val})

        # case 4
        a = tf.placeholder(tf.float32, shape=(3, 2))
        b = tf.placeholder(tf.float32, shape=(4, 3))
        a_val = np.random.rand(*tf_obj_shape(a))
        b_val = np.random.rand(*tf_obj_shape(b))
        self.run(tf.matmul(a, b, transpose_a=True, transpose_b=True),
                 tf_feed_dict={a: a_val, b: b_val})
