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

from __future__ import division

import pytest
import numpy as np
import ngraph as ng


def sequential():
    pass


@pytest.mark.skip(reason="why can't we add tests that fail?")
def test_init_gaussian():
    N = 128
    C = 4
    # XS, YS = g.gen_data(N, 10)

    X = ng.placeholder(axes=ng.Axes([C, N]))
    Y = ng.placeholder(axes=ng.Axes([N]))
    alpha = ng.placeholder(axes=ng.Axes())

#    W = ng.Variable(axes=ng.Axes([C]), initial_value=ng.fill(sequential))
    W = ng.Variable(axes=ng.Axes([C]), initial_value=10)

    L = W + 1 + alpha

    transformer = ng.NumPyTransformer()
    update_fun = transformer.computation([L, W], alpha, X, Y)

    xs = np.zeros((C, N), dtype=np.float32)
    ys = np.zeros((N,), dtype=np.float32)
    loss_val, w_val = update_fun(5.0, xs, ys)

    print(loss_val)
    transformer.close()
