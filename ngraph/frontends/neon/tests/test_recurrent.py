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
"""
This test compares the recurrent layer against a numpy reference recurrent
implementation.
The numpy reference recurrent layer contains static methods for forward pass
and backward pass.
The test runs a single layer of recurrent layer and compare numerical values
The reference model handles batch_size as 1 only

The following are made sure to be the same in both recurrent layers
    -   initial h values (all zeros)
    -   initial W, b (ones or random values)
    -   input data (random data matrix)
    -   input error (random data matrix)
    -   the data shape inside recurrent_ref is seq_len, input_size, 1
    -   the data shape inside recurrent (neon) is feature, seq_len * batch_size
"""
import itertools as itt

import numpy as np
from recurrent_ref import Recurrent as RefRecurrent

import ngraph as ng
from ngraph.frontends.neon import Recurrent, BiRNN, GaussianInit, Tanh
from ngraph.testing.execution import ExecutorFactory
from ngraph.testing.random import RandomTensorGenerator

rng = RandomTensorGenerator()

delta = 1e-3
rtol = atol = 1e-2


def pytest_generate_tests(metafunc):
    bsz_rng = [1]

    if 'rnn_args' in metafunc.fixturenames:
        seq_rng = [3]
        inp_rng = [5, 10]
        out_rng = [10, 32]
        ret_seq = [True, False]
        bwd = [True, False]
        fargs = itt.product(seq_rng, inp_rng, out_rng, bsz_rng, ret_seq, bwd)
        metafunc.parametrize('rnn_args', fargs)

    if 'birnn_args' in metafunc.fixturenames:
        seq_rng = [3]
        inp_rng = [5]
        out_rng = [10]
        ret_seq = [True, False]
        sum_out = [True, False]
        bi_fargs = itt.product(seq_rng, inp_rng, out_rng, bsz_rng, ret_seq, sum_out)
        metafunc.parametrize('birnn_args', bi_fargs)


def test_rnn_ones(rnn_args):
    # run comparison with reference code
    # for all ones init
    seq_len, input_size, hidden_size, batch_size, ret_seq, bwd = rnn_args
    check_rnn(seq_len, input_size, hidden_size,
              batch_size, (lambda x: 1.0), return_seq=ret_seq, backward=bwd)


def test_rnn_rand(rnn_args):
    # run comparison with reference code
    # for Gaussian random init
    seq_len, input_size, hidden_size, batch_size, ret_seq, bwd = rnn_args
    check_rnn(seq_len, input_size, hidden_size, batch_size,
              GaussianInit(0.0, 1.0), return_seq=ret_seq, backward=bwd)


def test_birnn(birnn_args):
    seq_len, input_size, hidden_size, batch_size, ret_seq, sum_out = birnn_args
    check_birnn(seq_len, input_size, hidden_size, batch_size,
                GaussianInit(0.0, 1.0), return_seq=ret_seq, sum_out=sum_out)


def check_birnn(seq_len, input_size, hidden_size, batch_size,
                init_func, return_seq, sum_out):
    # init_func is the initializer for the model params
    assert batch_size == 1, "the recurrent reference implementation only support batch size 1"

    # ========== neon model ==========
    Cin = ng.make_axis(input_size)
    REC = ng.make_axis(seq_len, recurrent=True)
    N = ng.make_axis(batch_size, batch=True)
    H = ng.make_axis(hidden_size)
    ax_s = ng.make_axes([H, N])

    with ExecutorFactory() as ex:
        np.random.seed(0)

        birnn_ng = BiRNN(hidden_size, init_func, activation=Tanh(),
                         reset_cells=True, return_sequence=return_seq, sum_out=sum_out)

        inp_ng = ng.placeholder([Cin, REC, N])
        init_state_ng = ng.placeholder(ax_s)

        # fprop graph
        out_ng = birnn_ng.train_outputs(inp_ng, init_state=init_state_ng)
        # out_ng.input = True

        rnn_W_input = birnn_ng.fwd_rnn.W_input
        rnn_W_input.input = True
        rnn_W_recur = birnn_ng.fwd_rnn.W_recur
        rnn_W_recur.input = True
        rnn_b = birnn_ng.fwd_rnn.b
        rnn_b.input = True

        # fprop on random inputs
        input_value = rng.uniform(-1, 1, inp_ng.axes)
        init_state_value = rng.uniform(-1, 1, init_state_ng.axes)

        if sum_out is True:
            fprop_neon_fun = ex.executor([out_ng,
                                      birnn_ng.fwd_rnn.W_input,
                                      birnn_ng.fwd_rnn.W_recur,
                                      birnn_ng.fwd_rnn.b,
                                      birnn_ng.bwd_rnn.W_input,
                                      birnn_ng.bwd_rnn.W_recur,
                                      birnn_ng.bwd_rnn.b],
                                      inp_ng, init_state_ng)
            fprop_neon, fwd_input, fwd_recur, fwd_b, bwd_input, bwd_recur, bwd_b = \
                fprop_neon_fun(input_value, init_state_value)
            fprop_neon = fprop_neon.copy()
        else:
            fprop_neon_fun = ex.executor(out_ng+
                                      [birnn_ng.fwd_rnn.W_input,
                                      birnn_ng.fwd_rnn.W_recur,
                                      birnn_ng.fwd_rnn.b,
                                      birnn_ng.bwd_rnn.W_input,
                                      birnn_ng.bwd_rnn.W_recur,
                                      birnn_ng.bwd_rnn.b],
                                      inp_ng, init_state_ng)
            fprop_neon, fprop_neon_1, fwd_input, fwd_recur, fwd_b, bwd_input, bwd_recur, bwd_b = \
                fprop_neon_fun(input_value, init_state_value)
            
            fprop_neon_fwd = fprop_neon.copy()
            fprop_neon_bwd = fprop_neon_1.copy()

        # ========= reference model ==========
        output_shape = (hidden_size, seq_len * batch_size)

        # generate random deltas tensor
        deltas = np.random.randn(*output_shape)

        # the reference code expects these shapes:
        # input_shape: (seq_len, input_size, batch_size)
        # output_shape: (seq_len, hidden_size, batch_size)
        deltas_ref = deltas.copy().T.reshape(
            seq_len, batch_size, hidden_size).swapaxes(1, 2)

        inp_ref = input_value.transpose([1, 0, 2])

        # reference numpy RNN
        rnn_ref = RefRecurrent(input_size, hidden_size, return_sequence=return_seq)
        rnn_ref.Wxh[:] = fwd_input.copy()
        rnn_ref.Whh[:] = fwd_recur.copy()
        rnn_ref.bh[:] = fwd_b.copy().reshape(rnn_ref.bh.shape)
        (dWxh_ref, dWhh_ref, db_ref, h_ref_fwd,
            dh_ref_list, d_out_ref) = rnn_ref.lossFun(inp_ref, deltas_ref,
                                                      init_states=init_state_value)

        rnn_ref.Wxh[:] = bwd_input.copy()
        rnn_ref.Whh[:] = bwd_recur.copy()
        rnn_ref.bh[:] = bwd_b.copy().reshape(rnn_ref.bh.shape)
        h_ref_bwd = rnn_ref.fprop_backwards(inp_ref, init_state_value)

        if sum_out is True:
            h_ref = h_ref_bwd + h_ref_fwd
            if return_seq is True:
                fprop_neon = fprop_neon[:, :, 0]
            ng.testing.assert_allclose(fprop_neon, h_ref, rtol=0.0, atol=1.0e-5)
        else:
            if return_seq is True:
                fprop_neon_fwd = fprop_neon_fwd[:, :, 0]
                fprop_neon_bwd = fprop_neon_bwd[:, :, 0]
            ng.testing.assert_allclose(fprop_neon_fwd, h_ref_fwd, rtol=0.0, atol=1.0e-5)
            ng.testing.assert_allclose(fprop_neon_bwd, h_ref_bwd, rtol=0.0, atol=1.0e-5)
        return


# compare neon RNN to reference RNN implementation
def check_rnn(seq_len, input_size, hidden_size, batch_size,
              init_func, return_seq=True, backward=False):
    # init_func is the initializer for the model params
    assert batch_size == 1, "the recurrent reference implementation only support batch size 1"

    # ========== neon model ==========
    Cin = ng.make_axis(input_size)
    REC = ng.make_axis(seq_len, recurrent=True)
    N = ng.make_axis(batch_size, batch=True)
    H = ng.make_axis(hidden_size)
    ax_s = ng.make_axes([H, N])

    with ExecutorFactory() as ex:
        np.random.seed(0)

        rnn_ng = Recurrent(hidden_size, init_func, activation=Tanh(),
                           reset_cells=True, return_sequence=return_seq,
                           backward=backward)

        inp_ng = ng.placeholder([Cin, REC, N])
        init_state_ng = ng.placeholder(ax_s)

        # fprop graph
        out_ng = rnn_ng.train_outputs(inp_ng, init_state=init_state_ng)
        out_ng.input = True

        rnn_W_input = rnn_ng.W_input
        rnn_W_input.input = True
        rnn_W_recur = rnn_ng.W_recur
        rnn_W_recur.input = True
        rnn_b = rnn_ng.b
        rnn_b.input = True

        fprop_neon_fun = ex.executor([out_ng, rnn_ng.W_input, rnn_ng.W_recur, rnn_ng.b], inp_ng, init_state_ng)

        dWrecur_s_fun = ex.derivative(out_ng, rnn_W_recur, inp_ng, rnn_W_input, rnn_b)
        dWrecur_n_fun = ex.numeric_derivative(out_ng, rnn_W_recur, delta, inp_ng, rnn_W_input, rnn_b)
        dWinput_s_fun = ex.derivative(out_ng, rnn_W_input, inp_ng, rnn_W_recur, rnn_b)
        dWinput_n_fun = ex.numeric_derivative(out_ng, rnn_W_input, delta, inp_ng, rnn_W_recur, rnn_b)
        dWb_s_fun = ex.derivative(out_ng, rnn_b, inp_ng, rnn_W_input, rnn_W_recur)
        dWb_n_fun = ex.numeric_derivative(out_ng, rnn_b, delta, inp_ng, rnn_W_input, rnn_W_recur)

        # fprop on random inputs
        input_value = rng.uniform(-1, 1, inp_ng.axes)
        init_state_value = rng.uniform(-1, 1, init_state_ng.axes)
        fprop_neon, Wxh_neon, Whh_neon, bh_neon  = fprop_neon_fun(input_value, init_state_value)
        fprop_neon = fprop_neon.copy()

        # after the rnn graph has been executed, can get the W values. Get copies so
        # shared values don't confuse derivatives
        #Wxh_neon = rnn_ng.W_input.value.get(None).copy()
        #Whh_neon = rnn_ng.W_recur.value.get(None).copy()
        #bh_neon = rnn_ng.b.value.get(None).copy()

        # bprop derivs
        dWrecur_s = dWrecur_s_fun(Whh_neon, input_value, Wxh_neon, bh_neon)
        dWrecur_n = dWrecur_n_fun(Whh_neon, input_value, Wxh_neon, bh_neon)
        ng.testing.assert_allclose(dWrecur_s, dWrecur_n, rtol=rtol, atol=atol)

        dWb_s = dWb_s_fun(bh_neon, input_value, Wxh_neon, Whh_neon)
        dWb_n = dWb_n_fun(bh_neon, input_value, Wxh_neon, Whh_neon)
        ng.testing.assert_allclose(dWb_s, dWb_n, rtol=rtol, atol=atol)

        dWinput_s = dWinput_s_fun(Wxh_neon, input_value, Whh_neon, bh_neon)
        dWinput_n = dWinput_n_fun(Wxh_neon, input_value, Whh_neon, bh_neon)
        ng.testing.assert_allclose(dWinput_s, dWinput_n, rtol=rtol, atol=atol)

        # ========= reference model ==========
        output_shape = (hidden_size, seq_len * batch_size)

        # generate random deltas tensor
        deltas = np.random.randn(*output_shape)

        # the reference code expects these shapes:
        # input_shape: (seq_len, input_size, batch_size)
        # output_shape: (seq_len, hidden_size, batch_size)
        deltas_ref = deltas.copy().T.reshape(
            seq_len, batch_size, hidden_size).swapaxes(1, 2)

        inp_ref = input_value.transpose([1, 0, 2])

        # reference numpy RNN
        rnn_ref = RefRecurrent(input_size, hidden_size, return_sequence=return_seq)
        rnn_ref.Wxh[:] = Wxh_neon
        rnn_ref.Whh[:] = Whh_neon
        rnn_ref.bh[:] = bh_neon.reshape(rnn_ref.bh.shape)

        if backward:
            h_ref_list = rnn_ref.fprop_backwards(inp_ref, init_state_value)
        else:
            (dWxh_ref, dWhh_ref, db_ref, h_ref_list,
                dh_ref_list, d_out_ref) = rnn_ref.lossFun(inp_ref, deltas_ref,
                                                          init_states=init_state_value)

        # comparing outputs
        if return_seq is True:
            fprop_neon = fprop_neon[:, :, 0]
        ng.testing.assert_allclose(fprop_neon, h_ref_list, rtol=0.0, atol=1.0e-5)

        return


if __name__ == '__main__':
    seq_len, input_size, hidden_size, batch_size = (3, 3, 6, 1)
    init = GaussianInit(0.0, 0.1)
    # check_rnn(seq_len, input_size, hidden_size, batch_size, init, False)
    check_birnn(seq_len, input_size, hidden_size, batch_size,
                init, return_seq=True, sum_out=True)
