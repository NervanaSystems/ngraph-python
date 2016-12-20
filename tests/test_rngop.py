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
"""
Test the usage of ng.constant
"""
from __future__ import print_function

import numpy as np

import ngraph as ng
from ngraph.testing import executor


def test_uniform_range_pos(transformer_factory):
    """TODO."""
    M = ng.make_axis(5, name='M')
    N = ng.make_axis(8, name='N')

    ng_a = ng.persistent_tensor([M, N], initial_value=10.0)
    ng_a = ng.uniform(ng_a, low=0.0, high=0.5)

    result = executor(ng_a)()
    print(result)

    assert np.all(result < 0.5)
    assert np.all(result >= 0.0)
    assert not np.all(result == 0.0)


def test_uniform_range_posneg(transformer_factory):
    """TODO."""
    M = ng.make_axis(5, name='M')
    N = ng.make_axis(8, name='N')

    ng_a = ng.persistent_tensor([M, N], initial_value=10.0)
    ng_a = ng.uniform(ng_a, low=-0.5, high=0.5)

    result = executor(ng_a)()
    print(result)

    assert np.all(result < 0.5)
    assert np.all(result >= -0.5)
    assert not np.all(result >= 0.0)
