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
This is a minimal single layer RNN implementation adapted from Andrej Karpathy's
code -- Minimal character-level Vanilla RNN model. BSD License
https://gist.github.com/karpathy/d4dee566867f8291f086

The adaptation includes
  - remove the file I/O
  - remove the recurrent to output affine layer
  - remove the sampling part
  - add a class container for the sizes and weights
  - keep only the lossFun function with provided inputs and errors
  - initialize weights and biases into empty, as the main test script will externally
    initialize the weights and biases
  - being able to read out hashable values to compare with another recurrent
    implementation
  - allow setting initial states
"""

import numpy as np
import collections


class RefRecurrent(object):

    def __init__(self, in_size, hidden_size, return_sequence=True):
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.Wxh = np.zeros((hidden_size, in_size))  # input to hidden
        self.Whh = np.zeros((hidden_size, hidden_size))  # hidden to hidden
        self.bh = np.zeros((hidden_size, 1))  # hidden bias
        self.return_sequence = return_sequence

    def set_weights(self, input_weights=None, recurrent_weights=None, bias=None):

        self.Wxh = input_weights if input_weights is not None else self.Wxh
        self.Whh = recurrent_weights if recurrent_weights is not None else self.Whh
        self.bh = bias if bias is not None else self.bh

    def lossFun(self, inputs, errors, init_states=None):
        """
        inputs,errors are both list of integers.
        returns the hidden states and gradients on model parameters
        """
        xs, hs = {}, {}
        if init_states is not None:
            assert init_states.shape == (self.hidden_size, 1)
            hs[-1] = init_states
        else:
            hs[-1] = np.zeros((self.hidden_size, 1))
        seq_len = len(inputs)
        hs_list = np.zeros((self.hidden_size, seq_len))
        nin = inputs[0].shape[0]

        # forward pass
        for t in range(seq_len):
            xs[t] = np.matrix(inputs[t])
            # hidden state
            hs[t] = np.tanh(
                np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)
            hs_list[:, t] = hs[t].flatten()

        hs_return = hs_list if self.return_sequence else hs_list[:, -1].reshape(-1, 1)

        # backward pass: compute gradients going backwards
        dhnext = np.zeros_like(hs[0])
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        dh_list = errors
        dh_list_out = np.zeros_like(dh_list)
        dout_list = np.zeros((nin, seq_len))

        for t in reversed(range(seq_len)):
            dh = dh_list[t] + dhnext  # backprop into h
            dh_list_out[t] = dh
            # backprop through tanh nonlinearity
            dhraw = np.multiply(dh, (1 - np.square(hs[t])))
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
            dout = np.dot(self.Wxh.T, dhraw)
            dout_list[:, t] = dout.flatten()

        return dWxh, dWhh, dbh, hs_return, dh_list_out, dout_list

    def fprop_only(self, inputs, backward=False, init_states=None):
        seq_len = len(inputs)

        init_idx = seq_len if backward is True else -1
        step_list = reversed(range(seq_len)) if backward is True else range(seq_len)
        last_idx = 0 if backward is True else -1

        xs, hs = {}, {}
        if init_states is not None:
            assert init_states.shape == (self.hidden_size, 1)
            hs[init_idx] = init_states
        else:
            hs[init_idx] = np.zeros((self.hidden_size, 1))

        hs_list = np.zeros((self.hidden_size, seq_len))

        for t in step_list:
            xs[t] = np.matrix(inputs[t])
            prev_idx = t + 1 if backward is True else t - 1
            # hidden state
            hs[t] = np.tanh(
                np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[prev_idx]) + self.bh)
            hs_list[:, t] = hs[t].flatten()

        hs_return = hs_list if self.return_sequence else hs_list[:, last_idx].reshape(-1, 1)

        return hs_return


class RefBidirectional(object):

    def __init__(self, in_size, hidden_size, return_sequence=True,
                 sum_out=False, concat_out=False):

        self.fwd_rnn = RefRecurrent(in_size, hidden_size, return_sequence=return_sequence)
        self.bwd_rnn = RefRecurrent(in_size, hidden_size, return_sequence=return_sequence)
        self.return_sequence = return_sequence
        self.sum_out = sum_out
        self.concat_out = concat_out

    def set_weights(self, input_weights=None, recurrent_weights=None, bias=None):

        self.fwd_rnn.set_weights(input_weights, recurrent_weights, bias)
        self.bwd_rnn.set_weights(input_weights, recurrent_weights, bias)

    def fprop(self, inputs, init_states=None):

        if isinstance(inputs, collections.Sequence):
            if len(inputs) != 2:
                raise ValueError("If in_obj is a sequence, it must have length 2")
            if inputs[0].axes != inputs[1].axes:
                raise ValueError("If in_obj is a sequence, each element must have the same axes")
            fwd_in = inputs[0]
            bwd_in = inputs[1]
        else:
            fwd_in = inputs
            bwd_in = inputs

        if isinstance(init_states, collections.Sequence):
            if len(init_states) != 2:
                raise ValueError("If init_state is a sequence, it must have length 2")
            if init_states[0].axes != init_states[1].axes:
                raise ValueError("If init_state is a sequence, " +
                                 "each element must have the same axes")
            fwd_init = init_states[0]
            bwd_init = init_states[1]
        else:
            fwd_init = init_states
            bwd_init = init_states

        fwd_out = self.fwd_rnn.fprop_only(fwd_in, init_states=fwd_init)
        bwd_out = self.bwd_rnn.fprop_only(bwd_in, init_states=bwd_init, backward=True)

        if self.sum_out:
            return fwd_out + bwd_out
        elif self.concat_out:
            return np.concatenate([fwd_out, bwd_out], 0)
        else:
            return [fwd_out, bwd_out]
