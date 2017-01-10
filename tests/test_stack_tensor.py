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

delta = 1e-3
rtol = atol = 1e-2


def test_stack(transformer_factory):
    ax = ng.make_name_scope(name="ax")
    ax.W = ng.make_axis(length=4)
    ax.H = ng.make_axis(length=5)
    ax.I = ng.make_axis(length=3)

    axes = ng.make_axes([ax.W, ax.H])

    rng = RandomTensorGenerator(0, np.float32)

    a_v = [rng.uniform(0, 1, axes) for i in range(ax.I.length)]

    for pos in range(len(axes) + 1):
        a = [ng.placeholder(axes, initial_value=_) for _ in a_v]

        s = ng.stack(a, ax.I, pos)

        ex = ExecutorFactory()

        num_funs = [ex.numeric_derivative(s, _, delta) for _ in a]
        sym_funs = [ex.derivative(s, _) for _ in a]

        ex.transformer.initialize()

        for n_fun, s_fun, a_i in zip(num_funs, sym_funs, a_v):
            d_n = n_fun(a_i)
            d_s = s_fun(a_i)
            ng.testing.allclose(d_n, d_s, rtol=rtol, atol=atol)
