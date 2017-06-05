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
import ngraph.transformers as ngt
from ngraph.testing import RandomTensorGenerator, ExecutorFactory
from ngraph.frontends.neon import ax
from third_party.warp_ctc.ctc import CTC
import pytest

pytestmark = [pytest.mark.transformer_dependent("module"),
              pytest.mark.flex_disabled("module")]


rng = RandomTensorGenerator(0, np.float32)


def pytest_generate_tests(metafunc):
    bsz = [1]

    if 'data_args' in metafunc.fixturenames:
        fargs = []
        nout = [5, 7, 9]
        bsz = [4, 32]
        max_utt_len = [23, 31, 39]
        max_lbl_len = [8, 9, 10]
        fargs = itt.product(nout, bsz, max_utt_len, max_lbl_len)
        metafunc.parametrize('data_args', fargs)


def ctc_ref(acts, lbls, utt_lens, lbl_lens):
    """
    CTC reference implementation
    """
    warp_ctc = CTC(on_device='cpu')
    max_t, bsz, nout = acts.shape
    grads = np.zeros_like(acts)
    costs = np.zeros(bsz, dtype=acts.dtype)
    utt_lens = (utt_lens * max_t / 100).astype(np.int32)
    warp_ctc.bind_to_cpu(acts,
                         lbls,
                         utt_lens,
                         lbl_lens,
                         grads,
                         costs,
                         n_threads=8)
    return costs, grads


def test_ctc(transformer_factory, data_args):
    """
    test ctc fprop and bprop
    """
    with ExecutorFactory() as ex:

        nout, bsz, max_utt_len, max_lbl_len = data_args
        V = ng.make_axis(nout)
        L = ng.make_axis(max_lbl_len * bsz)
        ax.N.length = bsz
        ax.REC.length = max_utt_len

        ax_activs = ng.make_axes([ax.REC, ax.N, V])
        ax_lbls = ng.make_axes([L])
        ax_utt_lens = ng.make_axes([ax.N])
        ax_lbl_lens = ng.make_axes([ax.N])

        activs = ng.placeholder(ax_activs)
        lbls = ng.placeholder(ax_lbls, dtype=np.dtype(np.int32))
        utt_lens = ng.placeholder(ax_utt_lens, dtype=np.dtype(np.int32))
        lbl_lens = ng.placeholder(ax_lbl_lens, dtype=np.dtype(np.int32))

        # fprop
        ctc_cost = ng.ctc(activs, lbls, utt_lens, lbl_lens)
        costfun = ex.executor(ctc_cost, activs, lbls, utt_lens, lbl_lens)

        # bprop
        grad_costfun = ex.derivative(ctc_cost, activs, lbls, utt_lens, lbl_lens)

        # provide numerical values and execute the graph
        activs_val = rng.uniform(-1, 1, activs.axes)
        lbls_val = np.random.randint(1, nout, lbls.axes.lengths, dtype=np.int32)
        lbl_lens_val = np.random.randint(1, max_lbl_len + 1,
                                         lbl_lens.axes.lengths, dtype=np.int32)
        utt_lens_val = ((2 * lbl_lens_val + 1) / float(max_utt_len)) * 100
        utt_lens_val = utt_lens_val.astype(np.int32)
        fprop_ctc = costfun(activs_val, lbls_val, utt_lens_val, lbl_lens_val).copy()
        bprop_ctc = grad_costfun(activs_val, lbls_val, utt_lens_val, lbl_lens_val).copy()

        # compare with reference values
        costs_ref, grads_ref = ctc_ref(activs_val, lbls_val, utt_lens_val, lbl_lens_val)
        ng.testing.assert_allclose(fprop_ctc, costs_ref, rtol=1.0e-5, atol=1.0e-5)
        ng.testing.assert_allclose(bprop_ctc[0], grads_ref, rtol=1.0e-5, atol=1.0e-5)


if __name__ == '__main__':
    factory = ngt.make_transformer_factory('cpu')
    ngt.set_transformer_factory(factory)
    bsz = 4
    nout = np.random.randint(5, 10)
    max_lbl_len = np.random.randint(5, 20)
    max_utt_len = 2 * max_lbl_len + 1
    data_args = (nout, bsz, max_utt_len, max_lbl_len)
    test_ctc(factory, data_args)
