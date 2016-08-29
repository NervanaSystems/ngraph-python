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
import geon as be
import numpy as np
from geon.util.utils import executor


def test_evalutaion_twice():
    """Test executing a computation graph twice on a one layer MLP."""
    x = be.Constant(
        np.array([[1, 2], [3, 4]], dtype='float32'),
        axes=be.Axes([be.NumericAxis(2), be.NumericAxis(2)])
    )

    hidden1_weights = be.Constant(
        np.array([[1], [1]], dtype='float32'),
        axes=be.Axes([be.NumericAxis(2), be.NumericAxis(1)])
    )

    hidden1_biases = be.Constant(
        np.array([[2], [2]], dtype='float32'),
        axes=be.Axes([be.NumericAxis(2), be.NumericAxis(1)])
    )

    hidden1 = be.dot(x, hidden1_weights) + hidden1_biases

    comp = executor(hidden1)

    result_1 = comp()
    result_2 = comp()
    assert np.array_equal(result_1, result_2)
