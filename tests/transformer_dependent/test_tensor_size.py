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
Test tensor size
On GPU, ng.tensor_size uses FillKernel
"""

import numpy as np
import ngraph as ng
from ngraph.testing import executor
import pytest


@pytest.mark.transformer_dependent
@pytest.config.argon_disabled  # TODO triage
def test_tensor_size():
    n, m = 3, 4

    N = ng.make_axis(length=n)
    M = ng.make_axis(length=m)

    aaxes = ng.make_axes([N, M])
    x = ng.placeholder(aaxes)

    size_fun = ng.tensor_size(x)
    nptensor = np.arange(n * m).reshape(n, m)

    with executor(size_fun, x) as ex:
        assert ex(nptensor) == n * m
