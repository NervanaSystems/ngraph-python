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
from tf_importer.tests.importer_tester import ImporterTester


class Tester(ImporterTester):
    def test_constant(self):
        # computation
        a = tf.constant(10.)
        b = tf.constant(20.)
        c = a + b
        d = c * a

        # test
        self.run(d)

    def test_constant_broadcast(self):
        # computation
        a = tf.constant([1, 2, 3.])
        b = tf.constant(20.)
        c = tf.ones([2, 3])
        d = tf.ones([10, 2, 3])
        e = tf.ones([1, 2, 3])
        f = a + b + c + d + e

        # test
        self.run(f)

    def test_fill(self):
        # computation
        f = tf.fill([2, 3], 5)

        # test
        self.run(f)

    def test_truncated_normal(self):
        # TODO
        pass

    def test_random_normal(self):
        # TODO
        pass
