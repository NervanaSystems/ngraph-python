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
import ngraph.transformers as ngt
from neon.initializers.initializer import Uniform
from neon import NervanaObject
from ngraph.frontends.neon.layer import Linear
from ngraph.util.utils import ExecutorFactory
from ngraph import RNG


def pytest_generate_tests(metafunc):

    bsz_rng = [128]

    if 'basic_linargs' in metafunc.fixturenames:
        nin_rng = [1, 2, 1023, 1024, 1025]
        nout_rng = [1, 4, 1023, 1024, 1025]
        fargs = itt.product(nin_rng, nout_rng, bsz_rng)
        metafunc.parametrize('basic_linargs', fargs)


def test_linear_zeros(basic_linargs, transformer_factory):
    ngt.make_transformer()
    NervanaObject.be.rng = RNG(0)
    # basic sanity check with 0 weights random inputs
    nin, nout, batch_size = basic_linargs
    init_unif = Uniform(low=0.0, high=0.0)

    # set inputs
    N = ng.make_axis("N", batch=True)
    F = ng.make_axis("F")
    N.length = batch_size
    F.length = nin

    inp = ng.placeholder(axes=ng.make_axes([F, N]))
    layer = Linear(nout=nout, init=init_unif)

    ex = ExecutorFactory()
    transformer = ex.transformer

    fprop = layer.configure(inp)

    # set up fprop
    output = transformer.computation(fprop, inp)

    # create data
    x = np.random.random((nin, batch_size))

    out = output(x)
    assert np.min(out) == 0.0 and np.max(out) == 0.0


def test_linear_ones(basic_linargs, transformer_factory):
    ngt.make_transformer()
    NervanaObject.be.rng = RNG(0)

    # basic sanity check with all ones on the inputs
    # and weights, check that each row in output
    # is the sum of the weights for that output
    # this check will confirm that the correct number
    # of operations is being run
    nin, nout, batch_size = basic_linargs
    init_unif = Uniform(low=1.0, high=1.0)

    # set inputs
    N = ng.make_axis("N", batch=True)
    F = ng.make_axis("F")
    N.length = batch_size
    F.length = nin

    inp = ng.placeholder(axes=ng.make_axes([F, N]))

    layer = Linear(nout=nout, init=init_unif)
    fprop = layer.configure(inp)
    transformer = ngt.make_transformer()

    # set up fprop
    output = transformer.computation(fprop, inp)

    # set up ability to retrieve weights
    weights = transformer.computation(layer.W)
    transformer.initialize()

    # run computation
    x = np.ones((nin, batch_size))

    out = output(x)
    w = weights()
    sums = np.sum(w, 1).reshape((nout, 1)) * np.ones((1, batch_size))

    assert np.allclose(sums, out, atol=0.0, rtol=0.0), \
        '%e' % np.max(np.abs(out - sums))
