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

import ngraph as ng
from ngraph.testing import ExecutorFactory, RandomTensorGenerator
import pytest


delta = 1e-3
rtol = atol = 1e-2


@pytest.config.flex_disabled(reason="Placeholders must be supplied - deriv problem?")
@pytest.config.argon_disabled  # TODO triage
@pytest.mark.transformer_dependent
def test_stack(transformer_factory):
    W = ng.make_axis(length=4)
    H = ng.make_axis(length=5)
    I = ng.make_axis(length=3)

    axes = ng.make_axes([W, H])

    rng = RandomTensorGenerator(0, np.float32)

    a_v = [rng.uniform(0, 1, axes) for i in range(I.length)]

    for pos in range(len(axes) + 1):
        a = [ng.placeholder(axes, initial_value=p) for p in a_v]

        s = ng.stack(a, I, pos)

        with ExecutorFactory() as ex:
            num_funs = [ex.numeric_derivative(s, p, delta, *(np for np in a if np is not p))
                        for p in a]
            sym_funs = [ex.derivative(s, p, *(np for np in a if np is not p))
                        for p in a]

            for n_fun, s_fun, a_i in zip(num_funs, sym_funs, a_v):
                na_is = list(na_i for na_i in a_v if na_i is not a_i)
                d_n = n_fun(a_i, *na_is)
                d_s = s_fun(a_i, *na_is)
            ng.testing.allclose(d_n, d_s, rtol=rtol, atol=atol)
