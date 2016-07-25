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
from geon.backends.graph.graph_test_utils import execute, be, Axes,\
    in_bound_environment
import numpy as np


@in_bound_environment
def test_dot_with_numerics():
    ax1 = be.NumericAxis(2)
    ax2 = be.NumericAxis(2)
    axes = Axes(ax1, ax2)

    x_np = np.array([[1, 2], [1, 2]], dtype='float32')
    x = be.NumPyTensor(x_np, axes=axes)

    d = be.dot(x, x, numpy_matching=True)
    d_val, = execute([d])

    assert np.array_equal(d_val, np.dot(x_np, x_np))

if __name__ == '__main__':
    test_dot_with_numerics()
