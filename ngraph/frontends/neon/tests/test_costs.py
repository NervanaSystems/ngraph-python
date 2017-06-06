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
import pytest
import numpy as np
import ngraph as ng
from ngraph.testing import executor

pytestmark = [pytest.mark.transformer_dependent, pytest.mark.flex_disabled("module")]


def safelog(x):
    return np.log(np.maximum(x, np.exp(-50)))


class CostPair(object):
    tolerance = 1e-6

    def reference_value(self, y, t):
        raise NotImplementedError("Must specify reference cost function")

    def baseline_value(self, y, t):
        '''
        Use defined ngraph constructed computation to evaluate
        cost function on inputs y and t
        '''
        N = ng.make_axis(length=y.shape[0])
        Y, T = ng.placeholder([N]), ng.placeholder([N])

        with executor(self.ng_computation(Y, T), Y, T) as ex:
            return ex(y, t)


class CrossEntropyBinaryPair(CostPair):
    def __init__(self):
        self.ng_computation = lambda Y, T: ng.cross_entropy_binary(Y, T)

    def reference_value(self, y, t):
        return np.sum((-t * safelog(y) - (1 - t) * safelog(1 - y)), keepdims=True)


class CrossEntropyMultiPair(CostPair):
    def __init__(self):
        self.ng_computation = lambda Y, T: ng.cross_entropy_multi(Y, T)

    def reference_value(self, y, t):
        return np.sum(-t * safelog(y), axis=0, keepdims=True)


class SumSquaredPair(CostPair):
    def __init__(self):
        self.ng_computation = lambda Y, T: ng.squared_L2(Y - T, out_axes=()) / 2

    def reference_value(self, y, t):
        return np.sum((y - t) ** 2, axis=0, keepdims=True) / 2


class MeanSquaredPair(CostPair):
    def __init__(self):
        self.ng_computation = lambda Y, T: ng.mean(ng.square(Y - T), out_axes=()) / 2.

    def reference_value(self, y, t):
        return np.mean((y - t) ** 2, axis=0, keepdims=True) / 2.


@pytest.mark.parametrize("y,t", [
    (np.array([0.5, 0.9, 0.1, 0.0001]), np.array([0.5, 0.99, 0.01, 0.2])),
    (np.array([0.5, 1.0, 0.0, 0.0001]), np.array([0.5, 0.0, 1.0, 0.2])),
])
@pytest.mark.parametrize("cost", [
    CrossEntropyMultiPair(),
    CrossEntropyBinaryPair(),
    SumSquaredPair(),
    MeanSquaredPair()
])
def test_costs(y, t, cost, transformer_factory):

    # X-FAIL for Flex's known issue
    if np.array_equal(y, np.array([0.5, 1.0, 0.0, 0.0001])) and \
       isinstance(cost, (CrossEntropyMultiPair, CrossEntropyBinaryPair)) and \
       transformer_factory.name == "flexgpu":
        pytest.xfail('Failing test for Flex: CrossEntropyMultiPair and CrossEntropyBinaryPair')

    ng.testing.assert_allclose(cost.baseline_value(y, t),
                               cost.reference_value(y, t),
                               rtol=cost.tolerance)
