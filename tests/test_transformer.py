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
import pytest

import ngraph as ng
from ngraph.testing import executor


def test_evalutaion_twice(transformer_factory):
    """Test executing a computation graph twice on a one layer MLP."""
    C = ng.make_axis(name='C')
    W = ng.make_axis(name='W')
    D = ng.make_axis(name='D')

    C.length = 2
    D.length = 2
    W.length = 1

    x = ng.constant(
        np.array([[1, 2], [3, 4]], dtype='float32'),
        ng.make_axes([C, D])
    )

    hidden1_weights = ng.constant(
        np.array([[1], [1]], dtype='float32'),
        ng.make_axes([C - 1, W])
    )

    hidden1_biases = ng.constant(
        np.array([[2], [2]], dtype='float32'),
        ng.make_axes([D, W])
    )

    hidden1 = ng.dot(hidden1_weights, x) + hidden1_biases

    with executor(hidden1) as comp:
        result_1 = comp()
        result_2 = comp()
    assert np.array_equal(result_1, result_2)


def test_missing_arguments_to_execute():
    """
    Expect a failure if the wrong number of arguments are passed to a
    computation.
    """
    N = ng.make_axis(1)

    x = ng.placeholder([N])
    y = ng.placeholder([N])

    with  executor(x + y, x, y) as f:
        with pytest.raises(ValueError):
            f(1)


def test_execute_non_placeholder():
    """
    Expect a failure if a non-input (Variable) is used as an argument to
    executor.
    """
    N = ng.make_axis(1)

    x = ng.variable([N])
    y = ng.variable([N])

    with pytest.raises(ValueError):
        with executor(x + y, x, y) as ex:
            ex
