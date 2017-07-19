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
from __future__ import division, print_function
from builtins import object

import collections
from contextlib import contextmanager
from cachetools import cached, keys
import ngraph as ng
from ngraph.frontends.neon.axis import shadow_axes_map, is_shadow_axis, reorder_spatial_axes
from ngraph.frontends.neon import Layer, get_steps, wrap_layer, LABELS

class Recurrent(Layer):
    """
    modified from basic recurrent layer to return both the last hidden state and sequence of hidden states

    Arguments:
        nout (int): Number of hidden/output units
        init (Initializer): Function for initializing the model's input to hidden weights.  By
                            default, this initializer will also be used for recurrent parameters
                            unless init_inner is also specified.  Biases will always be
                            initialized to zero.
        init_inner (Initializer, optional): Function for initializing the model's recurrent
                                            parameters.  If absent, will default to using same
                                            initializer provided to init.
        activation (Transform): Activation function for the input modulation
        batch_norm (bool, optional): defaults to False to not perform batch norm. If True,
                                     batch normalization is applied in each direction after
                                     multiplying the input by its W_input.
        reset_cells (bool): default to be True to make the layer stateless,
                            set to False to be stateful.
        return_sequence (bool): default to be True to return the whole sequence output.
        backward (bool): default to be False to process the sequence left to right
        name (str, optional): name to refer to this layer as.

    Attributes:
        W_input (Tensor): weights from inputs to output units
            (input_size, output_size)
        W_recur (Tensor): weights for recurrent connections
            (output_size, output_size)
        b (Tensor): Biases on output units (output_size, 1)
    """
    metadata = {'layer_type': 'recurrent'}

    def __init__(self, nout, init, init_inner=None, activation=None, batch_norm=False,
                 reset_cells=True, return_sequence=True, backward=False, **kwargs):
        super(Recurrent, self).__init__(**kwargs)

        self.nout = nout
        self.activation = activation
        self.init = init
        self.init_inner = init_inner if init_inner is not None else init
        self.reset_cells = reset_cells
        self.return_sequence = return_sequence
        self.backward = backward
        self.batch_norm = BatchNorm() if batch_norm is True else None
        self.w_in_axes = None

    def interpret_axes(self, in_obj, init_state):

        if self.w_in_axes is None:
            self.in_axes = in_obj.axes
            self.recurrent_axis = self.in_axes.recurrent_axis()
            self.in_feature_axes = self.in_axes.sample_axes() - self.recurrent_axis

            # if init state is given, use that as hidden axes
            if init_state:
                self.out_feature_axes = (init_state.axes.sample_axes() -
                                         init_state.axes.recurrent_axis())
                if sum(self.out_feature_axes.full_lengths) != self.nout:
                    raise ValueError("Length of init_state must be the same as nout: " +
                                     "{} != {}".format(sum(self.out_feature_axes.full_lengths),
                                                       self.nout))
            else:
                self.out_feature_axes = ng.make_axes([ng.make_axis(self.nout)])
                if len(self.in_feature_axes) == 1:
                    self.out_feature_axes[0].named(self.in_feature_axes[0].name)

            self.out_axes = self.out_feature_axes + self.in_axes.batch_axis()
            self.recurrent_axis_idx = len(self.out_feature_axes)

            # create temporary out axes which the dot ops will output.  These
            # temporary axes will be immediately cast to self.out_axes
            # afterwards.  We can't go directly to self.out_axes from the DotOp
            # because sometimes the self.out_axes intersect with the self.in_axes
            # and so the weight matrix would have a duplicate Axis which isn't
            # allowed.
            temp_out_axes = ng.make_axes(shadow_axes_map(self.out_feature_axes).keys())

            # determine the shape of the weight matrices
            self.w_in_axes = temp_out_axes + self.in_feature_axes
            self.w_re_axes = temp_out_axes + self.out_feature_axes

    def _step(self, h_ff, states):
        h_ff = ng.cast_role(h_ff, self.out_axes)
        h_rec = ng.cast_role(ng.dot(self.W_recur, states), self.out_axes)
        return self.activation(h_rec + h_ff + self.b)

    @ng.with_op_metadata
    @cached({}, key=Layer.inference_mode_key)
    def __call__(self, in_obj, init_state=None):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                 information for layer
            init_state (Tensor): object that provides initial state

        Returns:
            rnn_out (Tensor): output

        """
        # try to understand the axes from the input
        self.interpret_axes(in_obj, init_state)

        # initialize the hidden states
        if init_state is not None:
            self.h_init = init_state
        else:
            if self.reset_cells:
                self.h_init = ng.constant(
                    const=0, axes=self.out_axes).named('h_init')
            else:
                self.h_init = ng.variable(
                    initial_value=0, axes=self.out_axes).named('h_init')

        self.W_input = ng.variable(axes=self.w_in_axes,
                                   initial_value=self.init,
                                   scope=self.scope).named("W_in")
        self.W_recur = ng.variable(axes=self.w_re_axes,
                                   initial_value=self.init_inner,
                                   scope=self.scope).named("W_re")
        self.b = ng.variable(axes=self.out_feature_axes, initial_value=0,
                             scope=self.scope).named("bias")

        h = self.h_init
        h_list = []

        h_ff = ng.dot(self.W_input, in_obj)
        # Batch norm is computed only on the weighted inputs
        # as in https://arxiv.org/abs/1510.01378
        if self.batch_norm is not None:
            h_ff = self.batch_norm(h_ff)

        # slice the weighted inputs into time slices
        in_s = get_steps(h_ff, self.recurrent_axis, self.backward)

        # unrolling computations
        for i in range(self.recurrent_axis.length):
            with ng.metadata(recurrent_step=str(i)):
                h = self._step(in_s[i], h)
                h_list.append(h)

        if self.return_sequence is True:
            # only when returning a sequence, need to reverse the output
            h_list = h_list[::-1] if self.backward else h_list
            rnn_out = ng.stack(h_list, self.recurrent_axis, pos=self.recurrent_axis_idx)
            return h_list[-1], rnn_out
        else:
            # only return the last hidden
            return h_list[-1], _

class LSTM(Recurrent):
    """
    modified from LSTM layer to return both the last hidden state/cell and sequence of hidden states/cells

    Long Short-Term Memory (LSTM) layer based on
    Hochreiter and Schmidhuber, Neural Computation 9(8): 1735-80 (1997).

    Arguments:
        nout (int): Number of hidden/output units
        init (Initializer): Function for initializing the model's input to hidden weights.  By
                            default, this initializer will also be used for recurrent parameters
                            unless init_inner is also specified.  Biases will always be
                            initialized to zero.
        init_inner (Initializer, optional): Function for initializing the model's recurrent
                                            parameters.  If absent, will default to using same
                                            initializer provided to init.
        activation (Transform): Activation function for the input modulation
        batch_norm (bool, optional): defaults to False to not perform batch norm. If True,
                                     batch normalization is applied to each gate after
                                     multiplying the input by W_input.
        reset_cells (bool): default to be True to make the layer stateless,
                            set to False to be stateful.
        return_sequence (bool): default to be True to return the whole sequence output.
        backward (bool): default to be False to process the sequence left to right
        name (str, optional): name to refer to this layer as.
    Attributes:
        W_input (Tensor): weights from inputs to output units
            (output_size, input_size)
        W_recur (Tensor): weights for recurrent connections
            (output_size, output_size)
        b (Tensor): Biases on output units (output_size, 1)

    Gates: i - input gate, f - forget gate, o - output gate, g - input modulation
    """
    metadata = {'gates': ['i', 'f', 'o', 'g']}

    def __init__(self, nout, init, init_inner=None, activation=None, gate_activation=None,
                 batch_norm=False, reset_cells=True, return_sequence=True, backward=False,
                 **kwargs):
        super(LSTM, self).__init__(nout, init, init_inner=init_inner, activation=activation,
                                   reset_cells=reset_cells, return_sequence=return_sequence,
                                   backward=backward, **kwargs)

        if batch_norm is True:
            self.batch_norm = {k: BatchNorm() for k in self.metadata["gates"]}
        else:
            self.batch_norm = None
        self.gate_activation = gate_activation if gate_activation is not None else self.activation

    def _step(self, h_ff, states):
        h_state = states[0]
        c_state = states[1]
        ifog = {
            k: sum([ng.cast_role(h_ff[k], self.out_axes),
                    ng.cast_role(ng.dot(self.W_recur[k], h_state), self.out_axes),
                    self.b[k],
                    ]) for k in self.metadata['gates']
        }
        ifog_act = {k: self.activation(ifog[k]) if k is 'g'
                    else self.gate_activation(ifog[k]) for k in self.metadata['gates']}

        c = ifog_act['f'] * c_state + ifog_act['i'] * ifog_act['g']
        # c_prev is the state before applying activation
        h = ifog_act['o'] * self.activation(c)
        h = ng.cast_role(h, self.out_axes)
        return [h, c]

    @wrap_layer(cache_key=Layer.inference_mode_key)
    def __call__(self, in_obj, init_state=None, return_cell_state=False):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                 information for layer
            init_state (tuple of Tensor): object that provides initial state, and in LSTM,
                                          it includes hidden state, and cell states

        Returns:
            rnn_out (Tensor): output

        """

        # for seq2seq, recurrent axis and init_state change
        # between training and inference
        self.recurrent_axis = in_obj.axes.recurrent_axis()
        if init_state is not None:
            assert len(init_state) == 2 and init_state[0].axes == init_state[1].axes
            self.h_init = init_state[0]
            self.c_init = init_state[1]

        if not self.initialized:
            # try to understand the axes from the input
            if init_state is not None:
                self.interpret_axes(in_obj, init_state[0])
            else:
                self.interpret_axes(in_obj, init_state)

            # initialize the hidden states
            if init_state is None:
                if self.reset_cells:
                    self.h_init = ng.constant(const=0,
                                              axes=self.out_axes).named('h_init')
                    self.c_init = ng.constant(const=0,
                                              axes=self.out_axes).named('c_init')
                else:
                    self.h_init = ng.variable(initial_value=0,
                                              axes=self.out_axes).named('h_init')
                    self.c_init = ng.variable(initial_value=0,
                                              axes=self.out_axes).named('c_init')

            # params are dictionary for i, f, o, g
            gates = self.metadata["gates"]
            self.W_input = {k: ng.variable(axes=self.w_in_axes,
                                           initial_value=self.init,
                                           scope=self.scope,
                                           metadata={"label": LABELS["weight"]},
                                           ).named("W_in_{}".format(k)) for k in gates}

            self.W_recur = {k: ng.variable(axes=self.w_re_axes,
                                           initial_value=self.init_inner,
                                           scope=self.scope,
                                           metadata={"label": LABELS["weight"]},
                                           ).named("W_re_{}".format(k)) for k in gates}

            self.b = {k: ng.variable(axes=self.out_feature_axes,
                                     initial_value=0,
                                     scope=self.scope,
                                     metadata={"label": LABELS["bias"]},
                                     ).named("bias_{}".format(k)) for k in gates}

        h = self.h_init
        c = self.c_init

        h_list = []
        c_list = []

        # Compute feed forward weighted inputs
        # Batch norm is computed only on the weighted inputs
        # as in https://arxiv.org/abs/1510.01378
        h_ff = dict()
        for k in self.metadata["gates"]:
            h_ff[k] = ng.dot(self.W_input[k], in_obj)
            if self.batch_norm is not None:
                h_ff[k] = self.batch_norm[k](h_ff[k])

            # slice the weighted inputs into time slices
        h_ff = get_steps(h_ff, self.recurrent_axis, self.backward)

        # recurrent computation
        for i in range(self.recurrent_axis.length):
            with ng.metadata(recurrent_step=str(i)):
                [h, c] = self._step(h_ff[i], [h, c])
                h_list.append(h)
                c_list.append(c)

        if self.return_sequence is True:
            h_stack = ng.stack(h_list, self.recurrent_axis, pos=self.recurrent_axis_idx)
            if return_cell_state:
                c_stack = ng.stack(c_list, self.recurrent_axis, pos=self.recurrent_axis_idx)
                lstm_out = (h_stack, c_stack)
                return (h_list[-1], c_list[-1]), lstm_out
            else:
                return h_list[-1], h_stack
        else:
            if return_cell_state:
                return (h_list[-1], c_list[-1])
            else:
                return h_list[-1]
