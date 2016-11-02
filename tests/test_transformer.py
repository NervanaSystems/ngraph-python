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
import ngraph as ng
import numpy as np
import pytest
from ngraph.util.utils import executor


def test_evalutaion_twice(transformer_factory):
    """Test executing a computation graph twice on a one layer MLP."""
    C = ng.makeAxis(name='C')
    W = ng.makeAxis(name='W')
    D = ng.makeAxis(name='D')

    C.length = 2
    D.length = 2
    W.length = 1

    x = ng.Constant(
        np.array([[1, 2], [3, 4]], dtype='float32'),
        axes=ng.makeAxes([C, D])
    )

    hidden1_weights = ng.Constant(
        np.array([[1], [1]], dtype='float32'),
        axes=ng.makeAxes([C, W])
    )

    hidden1_biases = ng.Constant(
        np.array([[2], [2]], dtype='float32'),
        axes=ng.makeAxes([D, W])
    )

    hidden1 = ng.dot(x, hidden1_weights) + hidden1_biases

    comp = executor(hidden1)

    result_1 = comp()
    result_2 = comp()
    assert np.array_equal(result_1, result_2)


def test_missing_arguments_to_execute():
    """
    Expect a failure if the wrong number of arguments are passed to a
    computation.
    """
    N = ng.makeAxis(1)

    x = ng.placeholder(axes=[N])
    y = ng.placeholder(axes=[N])

    f = executor(x + y, x, y)
    with pytest.raises(ValueError):
        f(1)


def test_execute_non_placeholder():
    """
    Expect a failure if a non-input (Variable) is used as an argument to
    executor.
    """
    N = ng.makeAxis(1)

    x = ng.Variable(axes=[N])
    y = ng.Variable(axes=[N])

    with pytest.raises(ValueError):
        executor(x + y, x, y)
