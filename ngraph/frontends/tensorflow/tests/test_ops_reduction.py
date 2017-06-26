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
    @pytest.mark.parametrize("shape_and_reduction_indices", [
        [(), None],
        [(), ()],
        [(1,), None],
        [(1,), ()],
        [(1,), (0,)],
        [(3,), None],
        [(3,), ()],
        [(3,), (0,)],
        [(3, 4), (0,)],
        [(3, 4), (1,)],
        [(3, 4), (0, 1)],
        [(3, 4, 5), None],
        [(3, 4, 5), ()],
        [(3, 4, 5), (0,)],
        [(3, 4, 5), (0, 1)],
        [(3, 4, 5), (1, 2)],
        [(3, 4, 5), (0, 1, 2)]
    ])
    @pytest.mark.parametrize("op_name", [
        'reduce_sum',
        'reduce_prod',
        'reduce_mean'
    ])
    def test_reduction_ops(self, shape_and_reduction_indices, op_name):
        # test cases
        shape, reduction_indices = shape_and_reduction_indices
        tf_op = get_nested_attr(tf, op_name)

        # tf placeholder
        a = tf.placeholder(tf.float32, shape=shape)

        # value
        feed_dict = {a: np.random.rand(*tf_obj_shape(a))}

        # test
        f = tf_op(a, reduction_indices=reduction_indices)
        self.run(f, tf_feed_dict=feed_dict)
