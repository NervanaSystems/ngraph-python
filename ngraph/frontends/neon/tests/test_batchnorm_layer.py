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
import pytest
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import BatchNorm
from ngraph.testing.execution import ExecutorFactory


@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("input_size", [4, 10, 32])
@pytest.mark.parametrize("rho,epsilon", [(0.2, 0.1)])
def test_batchnorm_fprop(batch_size, input_size, rho, epsilon, transformer_factory):
    # This checks that that we are doing batch norm across a feature make_axis
    # and properly tracking the side effect variables
    np.random.seed(0)
    # set inputs
    N = ng.make_axis(batch_size, batch=True)
    F = ng.make_axis(input_size)

    input_placeholder = ng.placeholder([F, N])
    layer = BatchNorm(rho, epsilon)
    fprop = layer.train_outputs(input_placeholder)

    with ExecutorFactory() as ex:
        fprop_function = ex.transformer.computation(fprop, input_placeholder)
        stats_function = ex.transformer.computation([ng.value_of(layer.gmean),
                                                     ng.value_of(layer.gvar)])

        # initial conditions for tracked variables
        gmean_ref, gvar_ref = 0.0, 1.0

        # create data
        for i in range(2):
            x = np.random.random((input_size, batch_size)).astype(np.float32)

            out = fprop_function(x)
            gm, gv = stats_function()

            xmean = x.mean(axis=1, keepdims=True)
            xvar = x.var(axis=1, keepdims=True)
            out_ref = (x - xmean) / np.sqrt(xvar + epsilon)
            gmean_ref = xmean.ravel() * (1.0 - rho) + gmean_ref * rho
            gvar_ref = xvar.ravel() * (1.0 - rho) + gvar_ref * rho

            assert ng.testing.allclose(out,
                                       out_ref, atol=1e-6), '%e' % np.max(np.abs(out - out_ref))
            assert ng.testing.allclose(gm,
                                       gmean_ref, atol=1e-6), '%e' % np.max(np.abs(gm - gmean_ref))
            assert ng.testing.allclose(gv,
                                       gvar_ref, atol=1e-6), '%e' % np.max(np.abs(gv - gvar_ref))
