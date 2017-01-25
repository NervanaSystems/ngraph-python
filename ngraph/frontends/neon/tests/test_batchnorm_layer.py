# ----------------------------------------------------------------------------
# Copyright 2015-2017 Nervana Systems Inc.
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
'''
Test of the batchnorm layer
'''
import itertools as itt

import numpy as np

import ngraph as ng
from ngraph.frontends.neon import BatchNorm
from ngraph.testing.execution import ExecutorFactory


def pytest_generate_tests(metafunc):
    bsz_rng = [32, 64]

    if 'basic_bnargs' in metafunc.fixturenames:
        nin_rng = [4, 10, 32]
        fargs = itt.product(nin_rng, bsz_rng)
        metafunc.parametrize('basic_bnargs', fargs)


def test_batchnorm_fprop(basic_bnargs, transformer_factory):
    # This checks that that we are doing batch norm across a feature make_axis
    # and properly tracking the side effect variables
    nin, batch_size = basic_bnargs

    # set inputs
    N = ng.make_axis(batch_size, name="N", batch=True)
    F = ng.make_axis(nin, name="F")

    rho, eps = 0.2, 0.1
    inp = ng.placeholder([F, N])
    layer = BatchNorm(rho, eps)
    fprop = layer.train_outputs(inp)

    ex = ExecutorFactory()
    fprop_function = ex.executor([fprop, layer.gmean, layer.gvar], inp)
    np.random.seed(0)

    # initial conditions for tracked variables
    gmean_ref, gvar_ref = 0.0, 1.0

    # create data
    for i in range(2):
        x = np.random.random((nin, batch_size)).astype(np.float32)

        out, gm, gv = fprop_function(x)

        xmean = x.mean(axis=1, keepdims=True)
        xvar = x.var(axis=1, keepdims=True)
        out_ref = (x - xmean) / np.sqrt(xvar + eps)
        gmean_ref = xmean.ravel() * (1.0 - rho) + gmean_ref * rho
        gvar_ref = xvar.ravel() * (1.0 - rho) + gvar_ref * rho

        assert ng.testing.allclose(out, out_ref, atol=1e-6), '%e' % np.max(np.abs(out - out_ref))
        assert ng.testing.allclose(gm, gmean_ref, atol=1e-6), '%e' % np.max(np.abs(gm - gmean_ref))
        assert ng.testing.allclose(gv, gvar_ref, atol=1e-6), '%e' % np.max(np.abs(gv - gvar_ref))
