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
import itertools as itt
import numpy as np

import ngraph as ng
from ngraph.util.utils import RandomTensorGenerator, ExecutorFactory
from ngraph.frontends.neon import ax

rng = RandomTensorGenerator(0, np.float32)

delta = 1e-8
rtol = atol = 1e-2


def pytest_generate_tests(metafunc):
    bsz_rng = [1]

    if 'lut_args' in metafunc.fixturenames:
        fargs = []
        vocab_rng = [10, 50, 100]
        embed_rng = [20, 18, 31]
        bsz_rng = [4, 32]
        seq_rng = [3, 5]
        fargs = itt.product(vocab_rng, embed_rng, bsz_rng, seq_rng)
        metafunc.parametrize('lut_args', fargs)


def test_lut(lut_args):
    """
    test lut fprop and bprop
    """
    vocab_size, embed_dim, bsz, seq_len = lut_args

    V = ng.make_axis(vocab_size)
    F = ng.make_axis(embed_dim)
    ax.N.length = bsz
    ax.REC.length = seq_len

    ax_idx = ng.make_axes([ax.REC, ax.N])
    ax_lut = ng.make_axes([V, F])

    lut = ng.placeholder(ax_lut)
    idx = ng.placeholder(ax_idx)
    idx_flat = ng.flatten(idx)
    ax_out = idx_flat.axes + ng.make_axes([F])

    lut_out_ng = ng.lookuptable(lut, idx_flat, ax_out, pad_idx=0)

    ex = ExecutorFactory()

    # fprop
    fprop_fun = ex.executor(lut_out_ng, lut, idx)

    # bprop
    lut.input = True
    dW_lut_s_fun = ex.derivative(lut_out_ng, lut, idx)
    dW_lut_n_fun = ex.numeric_derivative(lut_out_ng, lut, delta, idx)

    # provide actual inputs and execute the graph
    lut_value = rng.uniform(-1, 1, lut.axes)
    idx_value = rng.random_integers(0, vocab_size - 1, idx.axes)

    fprop_lut = fprop_fun(lut_value, idx_value).copy()
    dlut_s = dW_lut_s_fun(lut_value, idx_value)
    dlut_n = dW_lut_n_fun(lut_value, idx_value)

if __name__ == '__main__':
    # lut_args = (10, 3, 4, 10)
    lut_args = (3, 3, 1, 1)
    test_lut(lut_args)
