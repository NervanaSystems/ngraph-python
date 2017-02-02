# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
This test compares the NEON LSTM layer against a numpy reference LSTM
implementation and compares the NEON LSTM bprop deltas to the gradients
estimated by finite differences.
The numpy reference LSTM contains static methods for forward pass
and backward pass.
It runs a SINGLE layer of LSTM and compare numerical values

The following are made sure to be the same in both LSTMs
    -   initial c, h values (all zeros)
    -   initial W, b (random values)
    -   input data (random data matrix)
    -   input error (random data matrix)
    -   the data shape inside LSTM_np is seq_len, batch_size, input_size.
        Need transpose
    -   the data shape inside LSTM (neon) is input_size, seq_len * batch_size

"""
import itertools as itt
import numpy as np
from lstm_ref import LSTM as RefLSTM

import ngraph as ng

from ngraph.frontends.neon import LSTM, GaussianInit, Tanh, Logistic
from ngraph.testing.execution import ExecutorFactory
from ngraph.testing.random import RandomTensorGenerator

rng = RandomTensorGenerator()

delta = 1e-3
rtol = atol = 1e-3


def pytest_generate_tests(metafunc):

    if 'reflstmargs' in metafunc.fixturenames:
        fargs = []
        bsz_rng = [1]
        seq_rng = [5, 10]
        inp_rng = [3, 5, 10]
        out_rng = [3, 5, 10]
        fargs = itt.product(seq_rng, inp_rng, out_rng, bsz_rng)
        metafunc.parametrize('reflstmargs', fargs)


def test_ref_compare_rand(transformer_factory, reflstmargs):
        # run comparison with reference code
        # for Gaussian random init
        seq_len, input_size, hidden_size, batch_size = reflstmargs
        check_lstm(seq_len, input_size, hidden_size, batch_size,
                   GaussianInit(0.0, 0.1))


def test_ref_stacked(transformer_factory, reflstmargs):
        seq_len, input_size, hidden_size, batch_size = reflstmargs
        check_stacked_lstm(seq_len, input_size, hidden_size, batch_size,
                           GaussianInit(0.0, 0.1))


# compare ngraph LSTM to reference LSTM implementation
def check_lstm(seq_len, input_size, hidden_size,
               batch_size, init_func, return_seq=True, backward=False,
               reset_cells=False):

    Cin = ng.make_axis(input_size)
    REC = ng.make_axis(seq_len, recurrent=True)
    N = ng.make_axis(batch_size, batch=True)

    with ExecutorFactory() as ex:
        np.random.seed(0)

        inp_ng = ng.placeholder([Cin, REC, N])

        lstm_ng = LSTM(hidden_size, init_func, activation=Tanh(), gate_activation=Logistic(),
                       reset_cells=reset_cells, return_sequence=return_seq,
                       backward=backward)

        out_ng = lstm_ng.train_outputs(inp_ng)

        fprop_neon_fun = ex.executor(out_ng, inp_ng)

        # fprop on random inputs
        input_value = rng.uniform(-1, 1, inp_ng.axes)
        fprop_neon = fprop_neon_fun(input_value).copy()

        # after the rnn graph has been executed, can get the W values. Get copies so
        # shared values don't confuse derivatives
        # concatenate weights to i, f, o, g together (in this order)
        gates = ['i', 'f', 'o', 'g']
        Wxh_neon = [lstm_ng.W_input[k].value.get(None).copy().T for k in gates]
        Whh_neon = [lstm_ng.W_recur[k].value.get(None).copy().T for k in gates]
        bh_neon = [lstm_ng.b[k].value.get(None).copy() for k in gates]

        # reference numpy LSTM
        lstm_ref = RefLSTM()
        WLSTM = lstm_ref.init(input_size, hidden_size)

        # make ref weights and biases with neon model
        WLSTM[0, :] = np.concatenate(bh_neon)
        WLSTM[1:input_size + 1, :] = np.concatenate(Wxh_neon, 1)
        WLSTM[input_size + 1:] = np.concatenate(Whh_neon, 1)

        # transpose input X and do fprop
        inp_ref = input_value.copy().transpose([1, 2, 0])
        (Hout_ref, cprev, hprev, batch_cache) = lstm_ref.forward(inp_ref,
                                                                 WLSTM)

        # the output needs transpose as well
        Hout_ref = Hout_ref.reshape(seq_len * batch_size, hidden_size).T

        # comparing outputs
        if return_seq is True:
            fprop_neon = fprop_neon[:, :, 0]

        assert ng.testing.allclose(fprop_neon, Hout_ref, rtol=rtol, atol=atol)


# compare ngraph LSTM to reference LSTM implementation
def check_stacked_lstm(seq_len, input_size, hidden_size,
                       batch_size, init_func, return_seq=True, backward=False,
                       reset_cells=True):

    Cin = ng.make_axis(input_size)
    REC = ng.make_axis(seq_len, recurrent=True)
    N = ng.make_axis(batch_size, batch=True)

    with ExecutorFactory() as ex:
        np.random.seed(0)

        inp_ng = ng.placeholder([Cin, REC, N])

        lstm_ng_1 = LSTM(hidden_size, init_func, activation=Tanh(), gate_activation=Logistic(),
                         reset_cells=reset_cells, return_sequence=return_seq,
                         backward=backward)
        lstm_ng_2 = LSTM(hidden_size, init_func, activation=Tanh(), gate_activation=Logistic(),
                         reset_cells=reset_cells, return_sequence=return_seq,
                         backward=backward)

        out_ng_1 = lstm_ng_1.train_outputs(inp_ng)
        out_ng_2 = lstm_ng_2.train_outputs(out_ng_1)

        fprop_neon_fun_1 = ex.executor(out_ng_1, inp_ng)
        fprop_neon_fun_2 = ex.executor(out_ng_2, inp_ng)

        # fprop on random inputs for multiple iterations
        fprop_neon_1_list = []
        fprop_neon_2_list = []
        input_value_list = []

        num_iter = 10
        for i in range(num_iter):
            input_value = rng.uniform(-1, 1, inp_ng.axes)
            fprop_neon_1 = fprop_neon_fun_1(input_value).copy()
            fprop_neon_2 = fprop_neon_fun_2(input_value).copy()

            # comparing outputs
            if return_seq is True:
                fprop_neon_1 = fprop_neon_1[:, :, 0]
                fprop_neon_2 = fprop_neon_2[:, :, 0]

            input_value_list.append(input_value)
            fprop_neon_1_list.append(fprop_neon_1)
            fprop_neon_2_list.append(fprop_neon_2)

        # after the rnn graph has been executed, can get the W values. Get copies so
        # shared values don't confuse derivatives
        # concatenate weights to i, f, o, g together (in this order)
        gates = ['i', 'f', 'o', 'g']
        Wxh_neon_1 = \
            np.concatenate([lstm_ng_1.W_input[k].value.get(None).copy().T for k in gates], 1)
        Whh_neon_1 = \
            np.concatenate([lstm_ng_1.W_recur[k].value.get(None).copy().T for k in gates], 1)
        bh_neon_1 =  \
            np.concatenate([lstm_ng_1.b[k].value.get(None).copy() for k in gates])
        Wxh_neon_2 = \
            np.concatenate([lstm_ng_2.W_input[k].value.get(None).copy().T for k in gates], 1)
        Whh_neon_2 = \
            np.concatenate([lstm_ng_2.W_recur[k].value.get(None).copy().T for k in gates], 1)
        bh_neon_2 = \
            np.concatenate([lstm_ng_2.b[k].value.get(None).copy() for k in gates])

        # reference numpy LSTM
        lstm_ref_1 = RefLSTM()
        lstm_ref_2 = RefLSTM()
        WLSTM_1 = lstm_ref_1.init(input_size, hidden_size)
        WLSTM_2 = lstm_ref_2.init(hidden_size, hidden_size)

        # make ref weights and biases the same with neon model
        WLSTM_1[0, :] = bh_neon_1
        WLSTM_1[1:input_size + 1, :] = Wxh_neon_1
        WLSTM_1[input_size + 1:] = Whh_neon_1
        WLSTM_2[0, :] = bh_neon_2
        WLSTM_2[1:hidden_size + 1, :] = Wxh_neon_2
        WLSTM_2[hidden_size + 1:] = Whh_neon_2

        # transpose input X and do fprop
        fprop_ref_1_list = []
        fprop_ref_2_list = []
        c0_1 = h0_1 = None
        c0_2 = h0_2 = None
        for i in range(num_iter):
            input_value = input_value_list[i]
            inp_ref = input_value.copy().transpose([1, 2, 0])
            (Hout_ref_1, cprev_1, hprev_1, batch_cache) = lstm_ref_1.forward(inp_ref, WLSTM_1,
                                                                             c0_1, h0_1)
            (Hout_ref_2, cprev_2, hprev_2, batch_cache) = lstm_ref_2.forward(Hout_ref_1, WLSTM_2,
                                                                             c0_2, h0_2)

            if reset_cells is False:
                c0_1 = cprev_1
                h0_1 = hprev_1
                c0_2 = cprev_2
                h0_2 = hprev_2

            # the output needs transpose as well
            Hout_ref_1 = Hout_ref_1.reshape(seq_len * batch_size, hidden_size).T
            Hout_ref_2 = Hout_ref_2.reshape(seq_len * batch_size, hidden_size).T

            fprop_ref_1_list.append(Hout_ref_1)
            fprop_ref_2_list.append(Hout_ref_2)

        for i in range(num_iter):
            assert ng.testing.allclose(fprop_neon_1_list[i],
                                       fprop_ref_1_list[i], rtol=rtol, atol=atol)
            assert ng.testing.allclose(fprop_neon_2_list[i],
                                       fprop_ref_2_list[i], rtol=rtol, atol=atol)


if __name__ == '__main__':
    seq_len, input_size, hidden_size, batch_size = (8, 5, 16, 1)
    init = GaussianInit(0.0, 0.1)
    check_lstm(seq_len, input_size, hidden_size, batch_size, init, return_seq=True)
