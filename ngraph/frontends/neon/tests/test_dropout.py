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
import numpy as np

import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.neon.layer import Dropout
from ngraph.testing.execution import executor


def test_dropout_train(transformer_factory):
    nin, batch_size = 32, 2

    # set inputs
    N = ng.make_axis(batch_size, batch=True).named('N')
    F = ng.make_axis(nin).named('F')

    inp = ng.placeholder([F, N])
    layer = Dropout(keep=0.5)
    fprop = layer.train_outputs(inp)

    # create data
    x = np.random.uniform(size=(nin, batch_size))

    # evaluate
    with executor([fprop, layer.mask], inp) as comp:
        out, mask = comp(x)
        numpy_out = x * mask[:, None]
        np.testing.assert_allclose(out, numpy_out, rtol=1e-6)

        out1, mask1 = out.copy(), mask.copy()
        out2, mask2 = comp(x)
        assert (out1 != out2).any()
        assert (mask1 != mask2).any()


def test_dropout_inference(transformer_factory):
    nin, batch_size = 8, 2

    # set inputs
    N = ng.make_axis(batch_size, batch=True).named('N')
    F = ng.make_axis(nin).named('F')

    inp = ng.placeholder([F, N])
    layer = Dropout(keep=0.5)
    fprop = layer.inference_outputs(inp)

    # create data
    x = np.random.uniform(size=(nin, batch_size))

    # evaluate
    with executor(fprop, inp) as comp:
        out = comp(x)
        numpy_out = x * 0.5
        np.testing.assert_allclose(out, numpy_out, rtol=1e-6)
        out1 = out.copy()
        out2 = comp(x)
        np.testing.assert_allclose(out1, out2, rtol=1e-6)


def test_dropout_bprop_single_comp(transformer_factory):
    nin, batch_size = 32, 2

    # set inputs
    N = ng.make_axis(batch_size, batch=True).named('N')
    F = ng.make_axis(nin).named('F')

    mul_factor = ng.placeholder(())
    inp = ng.placeholder([F, N])
    layer = Dropout(keep=0.5)
    fprop = layer.train_outputs(inp * mul_factor)
    out_graph = ng.sum(fprop, out_axes=())

    # create data
    x = np.random.uniform(size=(nin, batch_size))
    bprop = ng.deriv(out_graph, mul_factor)

    # evaluate
    trans = ngt.make_transformer()
    comp = trans.computation([fprop, bprop, layer.mask], inp, mul_factor)
    fout, bout, mask = comp(x, 2)
    # Calculate derivative by hand and compare
    np.testing.assert_allclose(bout, (x * mask[:, None]).sum(), rtol=1e-6)
    trans.cleanup()
