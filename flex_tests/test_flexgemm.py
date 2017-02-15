# -----------------------------------------------------------------------------
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
from ngraph.testing import executor, assert_allclose


# matrix multiply

def test_gemm(transformer_factory):
    """
    TODO: make this more interesting
    """
    n, c = 32, 32

    ax = ng.make_name_scope().named('ax')
    ax.N = ng.make_axis(length=n, name='N', batch=True)
    ax.C = ng.make_axis(length=c)

    X = ng.placeholder(axes=[ax.C, ax.N])
    Y = ng.placeholder(axes=[ax.N])

    W = ng.variable(axes=[ax.C - 1], initial_value=0.1)

    Y_hat = ng.dot(W, X)

    with executor(Y_hat, X) as ex:
        mm_executor = ex

        w = np.ones(c)*0.1
        xs = np.ones(n*c).reshape(c, n)

        for ii in range(3):
            y_hat_val = mm_executor(xs)
            # 8.8 fixed point test
            # assert np.allclose(np.dot(xs, w) - y_hat_val, 0.075*np.ones(n))

            # autoflex test
            assert_allclose(np.dot(xs, w), y_hat_val)
