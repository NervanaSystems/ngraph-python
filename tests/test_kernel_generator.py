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
from __future__ import division

import ngraph as ng
import ngraph.transformers as ngt
import numpy as np


def test_exit_condition(transformer_factory):
    bsz = 16
    class_num = 10

    N, Y = ng.make_axis(bsz), ng.make_axis(class_num)
    y_val = np.absolute(np.random.randn(bsz, class_num))
    y = ng.constant(y_val, ng.make_axes([N, Y]))

    likelihood = ng.log(ng.softmax(y, normalization_axes=y.axes[1]))

    transformer = ngt.make_transformer()
    comp = transformer.computation(likelihood)

    val1 = comp()
    val2 = comp()
    np.testing.assert_allclose(val1, val2, atol=0, rtol=0)
