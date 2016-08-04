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
import numpy as np

import geon as be
from geon.util.utils import execute, in_bound_environment


@in_bound_environment
def test_dot_with_numerics():
    ax1 = be.NumericAxis(2)
    ax2 = be.NumericAxis(2)
    axes = be.Axes(ax1, ax2)

    x_np = np.array([[1, 2], [1, 2]], dtype='float32')
    x = be.NumPyTensor(x_np, axes=axes)

    d = be.dot(x, x, numpy_matching=True)
    d_val, = execute([d])

    assert np.array_equal(d_val, np.dot(x_np, x_np))


@in_bound_environment
def test_expand_dims():
    ax1 = be.NumericAxis(2)
    ax2 = be.NumericAxis(2)
    axes = be.Axes(ax1, ax2)

    x_np = np.array([[1, 2], [1, 2]], dtype='float32')
    x = be.NumPyTensor(x_np, axes=axes)

    x1 = be.ExpandDims(x, ax1, 0)
    x1_val, = execute([x1])
    for i in range(ax1.length):
        assert np.array_equal(x1_val[i], x_np)
