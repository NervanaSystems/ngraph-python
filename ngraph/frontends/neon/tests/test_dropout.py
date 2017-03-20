# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
Test of the dropout layer
'''
import pytest
import numpy as np

from ngraph.testing import ExecutorFactory
import ngraph as ng
from ngraph.frontends.neon.layer import Layer, Dropout

pytestmark = [pytest.mark.transformer_dependent("module"),
              pytest.mark.flex_disabled("module")]

atol, rtol = 0, 1e-6


@pytest.mark.parametrize("nin,batch_size", [(32, 2)])
@pytest.mark.parametrize("keep", [1.0, 0.75, 0.5])
def test_dropout_train(nin, batch_size, keep, transformer_factory):

    # set inputs
    N = ng.make_axis(batch_size, name='N')
    F = ng.make_axis(nin, name='F')

    inp = ng.placeholder([F, N])
    layer = Dropout(keep=keep)
    fprop = layer(inp)

    # create data
    x = np.random.uniform(size=(nin, batch_size))

    # evaluate
    with ExecutorFactory() as ex:
        comp = ex.executor([fprop, layer.mask], inp)
        out, mask = comp(x)
        numpy_out = x * mask[:, None]
        ng.testing.assert_allclose(out, numpy_out, atol=atol, rtol=rtol)

        if keep < 1.0:
            out1, mask1 = out.copy(), mask.copy()
            out2, mask2 = comp(x)
            assert (out1 != out2).any()
            assert (mask1 != mask2).any()


@pytest.mark.parametrize("nin,batch_size", [(32, 2)])
def test_dropout_inference(nin, batch_size, transformer_factory):
    # set inputs
    N = ng.make_axis(batch_size, name='N')
    F = ng.make_axis(nin, name='F')

    inp = ng.placeholder([F, N])
    layer = Dropout(keep=0.5)
    with Layer.inference_mode_on():
        fprop = layer(inp)

    # create data
    x = np.random.uniform(size=(nin, batch_size))

    # evaluate
    with ExecutorFactory() as ex:
        comp = ex.executor(fprop, inp)
        out = comp(x)
        numpy_out = x * 0.5
        ng.testing.assert_allclose(out, numpy_out, atol=atol, rtol=rtol)
        out1 = out.copy()
        out2 = comp(x)
        ng.testing.assert_allclose(out1, out2, atol=atol, rtol=rtol)


@pytest.mark.parametrize("nin,batch_size", [(32, 2)])
@pytest.mark.parametrize("keep", [1.0, 0.5])
def test_dropout_bprop_single_comp(nin, batch_size, keep, transformer_factory):
    # set inputs
    N = ng.make_axis(batch_size, name='N')
    F = ng.make_axis(nin, name='F')

    mul_factor = ng.placeholder(())
    inp = ng.placeholder([F, N])
    layer = Dropout(keep=keep)
    fprop = layer(inp * mul_factor)
    out_graph = ng.sum(fprop, out_axes=())
    bprop = ng.deriv(out_graph, mul_factor)

    # create data
    x = np.random.uniform(size=(nin, batch_size))

    # evaluate
    with ExecutorFactory() as ex:
        comp = ex.executor([fprop, bprop, layer.mask], inp, mul_factor)
        fout, bout, mask = comp(x, 2)
        # Calculate derivative by hand and compare
        np.testing.assert_allclose(bout, (x * mask[:, None]).sum(), rtol=1e-6)
