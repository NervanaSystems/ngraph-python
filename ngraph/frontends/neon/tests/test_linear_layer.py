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
Test of the mlp/linear layer
'''
import itertools as itt
import numpy as np
import ngraph as ng
from neon.initializers.initializer import Uniform
from ngraph.frontends.neon import nnAffine, UniformInit
from ngraph.util.utils import ExecutorFactory
from ngraph.transformers import Transformer
from ngraph import RNG


def pytest_generate_tests(metafunc):

    bsz_rng = [128]

    if 'basic_linargs' in metafunc.fixturenames:
        nin_rng = [1, 2, 1023, 1024, 1025]
        nout_rng = [1, 4, 1023, 1024, 1025]
        fargs = itt.product(nin_rng, nout_rng, bsz_rng)
        metafunc.parametrize('basic_linargs', fargs)


def test_linear_zeros(basic_linargs, transformer_factory):

    # basic sanity check with 0 weights random inputs
    nin, nout, batch_size = basic_linargs

    # set inputs
    N = ng.Axis(batch_size, name="N", batch=True)
    F = ng.Axis(nin, name="F")
    O = ng.Axis(nout, name="out")

    inp = ng.placeholder(axes=ng.Axes([F, N]))
    layer = nnAffine(out_axis=O, init=UniformInit(0.0, 0.0))
    layer.initialize(ng.Axes([F]))
    fprop = layer.get_outputs(inp)

    ex = ExecutorFactory()
    transformer = ex.transformer
    output = transformer.computation(fprop, inp)
    transformer.initialize()

    # create data
    x = np.random.random((nin, batch_size))

    out = output(x)
    assert np.min(out) == 0.0 and np.max(out) == 0.0


def test_linear_ones(basic_linargs, transformer_factory):

    # basic sanity check with all ones on the inputs
    # and weights, check that each row in output
    # is the sum of the weights for that output
    # this check will confirm that the correct number
    # of operations is being run
    nin, nout, batch_size = basic_linargs

    # set inputs
    N = ng.Axis(batch_size, name="N", batch=True)
    F = ng.Axis(nin, name="F")
    O = ng.Axis(nout, name="out")

    inp = ng.placeholder(axes=ng.Axes([F, N]))
    layer = nnAffine(out_axis=O, init=UniformInit(0.0, 0.0))
    layer.initialize(ng.Axes([F]))
    fprop = layer.get_outputs(inp)

    ex = ExecutorFactory()
    transformer = ex.transformer
    output = transformer.computation(fprop, inp)
    weights = transformer.computation(layer.W)
    transformer.initialize()

    # create data
    x = np.ones((nin, batch_size))
    out = output(x)
    w = weights()

    sums = np.sum(w, 0).reshape((1, nout)) * np.ones((batch_size, 1))

    assert np.allclose(sums, out, atol=0.0, rtol=0.0), '%e' % np.max(np.abs(out - sums))
