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

import numpy as np
import pytest
import tensorflow as tf

from ngraph.frontends.tensorflow.tests.importer_tester import ImporterTester
from ngraph.frontends.tensorflow.tf_importer.utils import tf_obj_shape, \
    get_nested_attr

pytestmark = pytest.mark.transformer_dependent


# call by tf.unary_ew_op(tensor)
class Tester(ImporterTester):
    @pytest.mark.parametrize("op_name", [
        'tanh',
        'sigmoid',
        'nn.relu',
        'identity',
        'log',
        'neg',
        'square'
    ])
    def test_unary_ops(self, op_name):
        # op
        tf_op = get_nested_attr(tf, op_name)

        # input tensor
        x = tf.placeholder(tf.float32, shape=(3, 3))

        # feed_dict
        feed_dict = {x: np.random.rand(*tf_obj_shape(x))}

        # check tf vs ng
        f = tf_op(x)
        self.run(f, tf_feed_dict=feed_dict)
