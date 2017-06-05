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
import pytest


@pytest.mark.transformer_dependent
class Tester(ImporterTester):
    def test_constant(self):
        # computation
        a = tf.constant(10.)

        # test
        self.run(a)

    def test_fill(self):
        # computation
        f = tf.fill([2, 3], 5)

        # test
        self.run(f)

    def test_zeros_like(self):
        # computation
        a = tf.constant(np.ones((2, 3)).astype(np.float32))
        b = tf.zeros_like(a)

        # test
        self.run(b)

    def test_truncated_normal(self):
        # TODO
        pass

    def test_random_normal(self):
        # TODO
        pass
