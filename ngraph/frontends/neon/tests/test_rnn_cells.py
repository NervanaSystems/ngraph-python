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
This test compares the recurrent cells against a numpy reference recurrent
implementation.
The numpy reference recurrent layer contains static methods for forward pass
and backward pass.
The test runs a single layer of recurrent layer and compare numerical values
The reference model handles batch_size as 1 only

The following values are required to be the same for both recurrent layers
    -   initial h values (all zeros)
    -   initial W, b (ones or random values)
    -   input data (random data matrix)
    -   input error (random data matrix)
    -   the data shape inside recurrent_ref is seq_len, input_size, 1
    -   the data shape inside recurrent (neon) is feature, seq_len * batch_size
"""
import pytest

import numpy as np
from recurrent_ref import RefRecurrent

import ngraph as ng
from ngraph.frontends.neon import RNNCell, unroll, Tanh
from ngraph.testing.execution import ExecutorFactory
from ngraph.testing.random import RandomTensorGenerator


rng = RandomTensorGenerator()

delta = 1e-3
fprop_rtol = 0
fprop_atol = 1e-5
bprop_rtol = 0
bprop_atol = 1e-5

# numerical derivative is useful but shows large errors. Give high tolerance
num_atol = num_rtol = 1e-2

# TODO: Update tests to use conftest.py fixtures


@pytest.fixture(params=["random"])
def weight_initializer(request):
    if request.param == "random":
        return lambda w_axes: rng.normal(0, 1, w_axes)
    elif request.param == "ones":
        return lambda w_axes: np.ones(w_axes.lengths)


@pytest.fixture(params=["zeros"])
def bias_initializer(request):
    # TODO: Add useful bias init.
    # TODO: add more bias initializers for testing
    if request.param == "zeros":
        return lambda hidden_axis: np.zeros(hidden_axis.length)


def make_placeholder(input_size, sequence_length, batch_size, extra_axes=0):

    input_axis = ng.make_axis(name='features')
    recurrent_axis = ng.make_axis(name='REC')
    batch_axis = ng.make_axis(name='N')

    input_axes = ng.make_axes([input_axis, recurrent_axis, batch_axis])
    input_axes.set_shape((input_size, sequence_length, batch_size))
    input_axes = ng.make_axes([ng.make_axis(length=1, name='features_' + str(i))
                               for i in range(extra_axes)]) + input_axes

    input_placeholder = ng.placeholder(input_axes)
    input_value = rng.uniform(-0.01, 0.01, input_axes)

    return input_placeholder, input_value


def make_weights(input_placeholder, hidden_size, weight_initializer, bias_initializer,
                 init_state=False):
    in_feature_axes = tuple(input_placeholder.axes)[:-2]  # input axis + any extra axes of length 1
    out_feature_axes = ng.make_axes([ng.make_axis(hidden_size)])
    batch_axis = input_placeholder.axes.batch_axis()
    hidden_axis = ng.make_axis(hidden_size)

    w_in_axes = ng.make_axes(hidden_axis) + in_feature_axes
    w_rec_axes = ng.make_axes(hidden_axis) + out_feature_axes

    W_in = weight_initializer(w_in_axes)
    W_rec = weight_initializer(w_rec_axes)
    b = bias_initializer(hidden_axis)

    if init_state is True:
        ax_s = ng.make_axes([hidden_axis, batch_axis])
        init_state = ng.placeholder(ax_s)
        init_state_value = rng.uniform(-1, 1, ax_s)
    else:
        init_state = None
        init_state_value = None

    return W_in, W_rec, b, init_state, init_state_value


@pytest.config.argon_disabled  # TODO triage
@pytest.mark.transformer_dependent
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_length", [3])
@pytest.mark.parametrize("input_size", [5])
@pytest.mark.parametrize("hidden_size", [10])
@pytest.mark.parametrize("return_sequence", [True, False])
@pytest.mark.parametrize("init_state", [True, False])
@pytest.mark.parametrize("extra_axes", [0, 2])
@pytest.mark.parametrize("backward", [True, False])
def test_rnn_fprop(sequence_length, input_size, hidden_size, batch_size, return_sequence,
                   weight_initializer, bias_initializer, init_state, extra_axes, backward):

    assert batch_size == 1, "the recurrent reference implementation only support batch size 1"

    # Get input placeholder and numpy array
    input_placeholder, input_value = make_placeholder(input_size, sequence_length, batch_size,
                                                      extra_axes=extra_axes)

    # Construct network weights and initial state, if desired
    W_in, W_rec, b, init_state, init_state_value = make_weights(input_placeholder, hidden_size,
                                                                weight_initializer,
                                                                bias_initializer,
                                                                init_state)

    # Compute reference numpy RNN
    rnn_ref = RefRecurrent(input_size, hidden_size, return_sequence=return_sequence)
    rnn_ref.set_weights(W_in.reshape(rnn_ref.Wxh.shape), W_rec, b.reshape(rnn_ref.bh.shape))

    # Compute reference numpy RNN
    input_shape = (input_size, sequence_length, batch_size)
    h_ref_list = rnn_ref.fprop_only(input_value.reshape(input_shape).transpose([1, 0, 2]),
                                    init_states=init_state_value, backward=backward)

    # Generate ngraph RNN
    rnn_ng = RNNCell(hidden_size, init=W_in, init_h2h=W_rec, activation=Tanh(),
                     reset_cells=True)

    # fprop ngraph RNN
    num_steps = input_placeholder.axes.recurrent_axis().length
    init_states = {'h': init_state} if init_state is not None else init_state
    out_ng = unroll(rnn_ng, num_steps, input_placeholder, init_states=init_states,
                    return_sequence=return_sequence, reverse_mode=backward)

    with ExecutorFactory() as ex:
        # Create computation and execute
        if init_state is not None:
            fprop_neon_fun = ex.executor(out_ng, input_placeholder, init_state)
            fprop_neon = fprop_neon_fun(input_value, init_state_value)

        else:
            fprop_neon_fun = ex.executor(out_ng, input_placeholder)
            fprop_neon = fprop_neon_fun(input_value)

        # Compare output with reference implementation
        if return_sequence is True:
            fprop_neon = fprop_neon[:, :, 0]
        ng.testing.assert_allclose(fprop_neon, h_ref_list, rtol=fprop_rtol, atol=fprop_atol)


@pytest.config.flex_disabled(reason="RNN is not yet supported with Flex")
@pytest.config.argon_disabled  # TODO triage
@pytest.mark.transformer_dependent
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_length", [3])
@pytest.mark.parametrize("input_size", [5])
@pytest.mark.parametrize("hidden_size", [10])
@pytest.mark.parametrize("return_sequence", [True])
@pytest.mark.parametrize("init_state", [True, False])
def test_rnn_deriv_ref(sequence_length, input_size, hidden_size, batch_size, return_sequence,
                       weight_initializer, bias_initializer, init_state):

    assert batch_size == 1, "the recurrent reference implementation only support batch size 1"
    assert return_sequence is True, "the reference rnn only supports sequences for deriv"

    # Get input placeholder and numpy array
    input_placeholder, input_value = make_placeholder(input_size, sequence_length, batch_size)

    # Construct network weights and initial state, if desired
    W_in, W_rec, b, init_state, init_state_value = make_weights(input_placeholder, hidden_size,
                                                                weight_initializer,
                                                                bias_initializer,
                                                                init_state)

    # Compute reference numpy RNN
    rnn_ref = RefRecurrent(input_size, hidden_size, return_sequence=return_sequence)
    rnn_ref.set_weights(W_in, W_rec, b.reshape(rnn_ref.bh.shape))

    # Prepare deltas for gradient check
    output_shape = (hidden_size, sequence_length, batch_size)

    # generate random deltas tensor
    deltas = np.random.randn(*output_shape)

    # the reference code expects these shapes:
    # input_shape: (seq_len, input_size, batch_size)
    # output_shape: (seq_len, hidden_size, batch_size)
    dW_in, dW_rec, db = rnn_ref.lossFun(input_value.transpose([1, 0, 2]),
                                        deltas.copy().transpose([1, 0, 2]),
                                        init_states=init_state_value)[:3]

    # Generate ngraph RNN
    rnn_ng = RNNCell(hidden_size, init=W_in, init_h2h=W_rec, activation=Tanh(),
                     reset_cells=True)

    # fprop ngraph RNN
    num_steps = input_placeholder.axes.recurrent_axis().length
    init_states = {'h': init_state} if init_state is not None else init_state
    out_ng = unroll(rnn_ng, num_steps, input_placeholder, init_states=init_states,
                    return_sequence=return_sequence)
    deltas_constant = ng.constant(deltas, axes=out_ng.axes)
    params = [(rnn_ng.i2h.linear.W, W_in),
              (rnn_ng.h2h.W, W_rec),
              (rnn_ng.i2h.bias.W, b)]

    with ExecutorFactory() as ex:
        # Create derivative computations and execute
        param_updates = list()
        for px, _ in params:
            update = ng.deriv(out_ng, px, error=deltas_constant)
            if init_state is not None:
                param_updates.append(ex.executor(update, input_placeholder, init_state))
            else:
                param_updates.append(ex.executor(update, input_placeholder))

        for update_fun, ref_val in zip(param_updates, [dW_in, dW_rec, db]):
            if init_state is not None:
                grad_neon = update_fun(input_value, init_state_value)
            else:
                grad_neon = update_fun(input_value)
            ng.testing.assert_allclose(grad_neon,
                                       ref_val.squeeze(),
                                       rtol=bprop_rtol, atol=bprop_atol)


@pytest.config.flex_disabled(reason="Several: Tensor description, placeholder (deriv), tolerance")
@pytest.config.argon_disabled  # TODO triage
@pytest.mark.transformer_dependent
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_length", [3])
@pytest.mark.parametrize("input_size", [5])
@pytest.mark.parametrize("hidden_size", [10])
@pytest.mark.parametrize("return_sequence", [True, False])
@pytest.mark.parametrize("backward", [True, False])
@pytest.mark.parametrize("init_state", [True, False])
def test_rnn_deriv_numerical(sequence_length, input_size, hidden_size, batch_size, return_sequence,
                             weight_initializer, bias_initializer, backward, init_state):

    # Get input placeholder and numpy array
    input_placeholder, input_value = make_placeholder(input_size, sequence_length, batch_size)

    # Construct network weights and initial state, if desired
    W_in, W_rec, b, init_state, init_state_value = make_weights(input_placeholder, hidden_size,
                                                                weight_initializer,
                                                                bias_initializer,
                                                                init_state)

    # Generate ngraph RNN
    rnn_ng = RNNCell(hidden_size, init=W_in, init_h2h=W_rec, activation=Tanh(),
                     reset_cells=True)

    # fprop ngraph RNN
    num_steps = input_placeholder.axes.recurrent_axis().length
    init_states = {'h': init_state} if init_state is not None else init_state
    out_ng = unroll(rnn_ng, num_steps, input_placeholder, init_states=init_states,
                    return_sequence=return_sequence)

    params = [(rnn_ng.i2h.linear.W, W_in),
              (rnn_ng.h2h.W, W_rec),
              # (rnn_ng.i2h.bias.W, b)
              ]

    with ExecutorFactory() as ex:
        # Create derivative computations and execute
        param_updates = list()
        for px, _ in params:
            if init_state is not None:
                update = (ex.derivative(out_ng, px, input_placeholder, init_state),
                          ex.numeric_derivative(out_ng, px, delta, input_placeholder, init_state))
            else:
                update = (ex.derivative(out_ng, px, input_placeholder),
                          ex.numeric_derivative(out_ng, px, delta, input_placeholder))
            param_updates.append(update)

        for (deriv_s, deriv_n), (_, val) in zip(param_updates, params):
            if init_state is not None:
                ng.testing.assert_allclose(deriv_s(val, input_value, init_state_value),
                                           deriv_n(val, input_value, init_state_value),
                                           rtol=num_rtol, atol=num_atol)
            else:
                ng.testing.assert_allclose(deriv_s(val, input_value),
                                           deriv_n(val, input_value),
                                           rtol=num_rtol, atol=num_atol)
