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
from ngraph.frontends.tensorflow.tf_importer.utils import np_layout_shuffle


def test_np_layout_shuffle():
    # set up
    bsz = 8
    C, H, W, N = 3, 28, 28, bsz
    C, R, S, K = 3, 5, 5, 32

    # image dim-shuffle
    np_tf_image = np.random.randn(N, H, W, C)
    np_ng_image = np_layout_shuffle(np_tf_image, "NHWC", "CDHWN")
    np_tf_image_reverse = np_layout_shuffle(np_ng_image, "CDHWN", "NHWC")
    assert np.array_equal(np_tf_image, np_tf_image_reverse)

    # filter dim-shuffle
    np_tf_weight = np.random.randn(R, S, C, K)
    np_ng_weight = np_layout_shuffle(np_tf_weight, "RSCK", "CTRSK")
    np_tf_weight_reverse = np_layout_shuffle(np_ng_weight, "CTRSK", "RSCK")
    assert np.array_equal(np_tf_weight, np_tf_weight_reverse)
