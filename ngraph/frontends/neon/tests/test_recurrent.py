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
import pytest

import numpy as np
from recurrent_ref import RefRecurrent, RefBidirectional, RefSeq2Seq

import ngraph as ng
from ngraph.frontends.neon import Recurrent, BiRNN, Tanh
from ngraph.testing.execution import ExecutorFactory
from ngraph.testing.random import RandomTensorGenerator

pytestmark = pytest.mark.transformer_dependent


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
    # For now, bias is initialized to 0 since there is not control within the RNN layer
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


@pytest.mark.flex_disabled
@pytest.config.argon_disabled  # TODO triage
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_length", [3])
@pytest.mark.parametrize("input_size", [5])
@pytest.mark.parametrize("hidden_size", [10])
@pytest.mark.parametrize("return_sequence", [True, False])
@pytest.mark.parametrize("init_state", [True, False])
@pytest.mark.parametrize("extra_axes", [0, 2])
@pytest.mark.parametrize("backward", [True, False])
def test_rnn_fprop(sequence_length, input_size, hidden_size, batch_size,
                   return_sequence, weight_initializer, bias_initializer,
                   init_state, extra_axes, backward, transformer_factory):

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
    rnn_ng = Recurrent(hidden_size, init=W_in, init_inner=W_rec, activation=Tanh(),
                       reset_cells=True, return_sequence=return_sequence,
                       backward=backward)

    # fprop ngraph RNN
    out_ng = rnn_ng(input_placeholder, init_state=init_state)

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


@pytest.mark.flex_disabled
@pytest.config.argon_disabled  # TODO triage
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_length", [3])
@pytest.mark.parametrize("input_size", [5])
@pytest.mark.parametrize("hidden_size", [10])
@pytest.mark.parametrize("return_sequence", [True])
@pytest.mark.parametrize("init_state", [True, False])
def test_rnn_deriv_ref(sequence_length, input_size, hidden_size, batch_size,
                       return_sequence, weight_initializer, bias_initializer,
                       init_state, transformer_factory):

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
    rnn_ng = Recurrent(hidden_size, init=W_in, init_inner=W_rec, activation=Tanh(),
                       reset_cells=True, return_sequence=return_sequence)

    # fprop ngraph RNN
    out_ng = rnn_ng(input_placeholder, init_state=init_state)

    deltas_constant = ng.constant(deltas, axes=out_ng.axes)
    params = [(rnn_ng.W_input, W_in),
              (rnn_ng.W_recur, W_rec),
              (rnn_ng.b, b)]

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


@pytest.mark.flex_disabled
@pytest.config.argon_disabled  # TODO triage
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_length", [3])
@pytest.mark.parametrize("input_size", [5])
@pytest.mark.parametrize("hidden_size", [10])
@pytest.mark.parametrize("return_sequence", [True, False])
@pytest.mark.parametrize("backward", [True, False])
@pytest.mark.parametrize("init_state", [True, False])
def test_rnn_deriv_numerical(sequence_length, input_size, hidden_size, batch_size,
                             return_sequence, weight_initializer, bias_initializer,
                             backward, init_state, transformer_factory):

    # Get input placeholder and numpy array
    input_placeholder, input_value = make_placeholder(input_size, sequence_length, batch_size)

    # Construct network weights and initial state, if desired
    W_in, W_rec, b, init_state, init_state_value = make_weights(input_placeholder, hidden_size,
                                                                weight_initializer,
                                                                bias_initializer,
                                                                init_state)

    # Generate ngraph RNN
    rnn_ng = Recurrent(hidden_size, init=W_in, init_inner=W_rec, activation=Tanh(),
                       reset_cells=True, return_sequence=return_sequence,
                       backward=backward)

    # fprop ngraph RNN
    out_ng = rnn_ng(input_placeholder, init_state=init_state)

    params = [(rnn_ng.W_input, W_in),
              (rnn_ng.W_recur, W_rec),
              (rnn_ng.b, b)]

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


@pytest.mark.flex_disabled
@pytest.config.argon_disabled  # TODO triage
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_length", [3])
@pytest.mark.parametrize("input_size", [5])
@pytest.mark.parametrize("hidden_size", [10])
@pytest.mark.parametrize("return_sequence", [True, False])
@pytest.mark.parametrize("init_state", [True, False])
@pytest.mark.parametrize("sum_out,concat_out", [(False, False),
                                                (True, False),
                                                (False, True)])
def test_birnn_fprop(sequence_length, input_size, hidden_size, batch_size,
                     return_sequence, weight_initializer, bias_initializer,
                     init_state, sum_out, concat_out, transformer_factory):

    assert batch_size == 1, "the recurrent reference implementation only support batch size 1"

    # Get input placeholder and numpy array
    input_placeholder, input_value = make_placeholder(input_size, sequence_length, batch_size)

    # Construct network weights and initial state, if desired
    W_in, W_rec, b, init_state, init_state_value = make_weights(input_placeholder, hidden_size,
                                                                weight_initializer,
                                                                bias_initializer,
                                                                init_state)

    # Compute reference numpy RNN
    rnn_ref = RefBidirectional(input_size, hidden_size, return_sequence=return_sequence,
                               sum_out=sum_out, concat_out=concat_out)
    rnn_ref.set_weights(W_in, W_rec, b.reshape(rnn_ref.fwd_rnn.bh.shape))
    h_ref_list = rnn_ref.fprop(input_value.transpose([1, 0, 2]),
                               init_states=init_state_value)

    # Generate ngraph RNN
    rnn_ng = BiRNN(hidden_size, init=W_in, init_inner=W_rec, activation=Tanh(),
                   reset_cells=True, return_sequence=return_sequence,
                   sum_out=sum_out, concat_out=concat_out)

    # fprop ngraph RNN
    out_ng = rnn_ng(input_placeholder, init_state=init_state)

    with ExecutorFactory() as ex:
        # Create computation and execute
        if init_state is not None:
            fprop_neon_fun = ex.executor(out_ng, input_placeholder, init_state)
            fprop_neon = fprop_neon_fun(input_value, init_state_value)

        else:
            fprop_neon_fun = ex.executor(out_ng, input_placeholder)
            fprop_neon = fprop_neon_fun(input_value)

        # Compare output with reference implementation
        if not isinstance(fprop_neon, tuple):
            fprop_neon = [fprop_neon]
            h_ref_list = [h_ref_list]
        for ii, output in enumerate(fprop_neon):
            if return_sequence is True:
                output = output[:, :, 0]
            ng.testing.assert_allclose(output, h_ref_list[ii], rtol=fprop_rtol, atol=fprop_atol)


@pytest.config.argon_disabled  # TODO triage
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_length", [3])
@pytest.mark.parametrize("input_size", [5])
@pytest.mark.parametrize("hidden_size", [10])
@pytest.mark.parametrize("return_sequence", [False, True])
@pytest.mark.parametrize("sum_out,concat_out", [(False, False),
                                                (True, False),
                                                (False, True)])
def test_birnn_deriv_numerical(sequence_length, input_size, hidden_size, batch_size,
                               return_sequence, weight_initializer, bias_initializer,
                               sum_out, concat_out):

    # Get input placeholder and numpy array
    input_placeholder, input_value = make_placeholder(input_size, sequence_length, batch_size)

    # Construct network weights and initial state, if desired
    W_in, W_rec, b, init_state, init_state_value = make_weights(input_placeholder, hidden_size,
                                                                weight_initializer,
                                                                bias_initializer)

    # Generate ngraph RNN
    rnn_ng = BiRNN(hidden_size, init=W_in, init_inner=W_rec, activation=Tanh(),
                   reset_cells=True, return_sequence=return_sequence,
                   sum_out=sum_out, concat_out=concat_out)

    # fprop ngraph RNN
    out_ng = rnn_ng(input_placeholder)

    w_in_f = rnn_ng.fwd_rnn.W_input
    w_rec_f = rnn_ng.fwd_rnn.W_recur
    b_f = rnn_ng.fwd_rnn.b
    w_in_b = rnn_ng.bwd_rnn.W_input
    w_rec_b = rnn_ng.bwd_rnn.W_recur
    b_b = rnn_ng.bwd_rnn.b

    params_f = [(w_in_f, W_in),
                (w_rec_f, W_rec),
                (b_f, b)]

    params_b = [(w_in_b, W_in),
                (w_rec_b, W_rec),
                (b_b, b)]

    if sum_out or concat_out:
        out_ng = [out_ng]
        params_birnn = [params_f + params_b]
    else:
        # in this case out_ng will be a list
        params_birnn = [params_f, params_b]

    with ExecutorFactory() as ex:
        # Create derivative computations and execute
        param_updates = list()
        dep_list = list()
        for output, dependents in zip(out_ng, params_birnn):
            for px, _ in dependents:
                update = (ex.derivative(output, px, input_placeholder),
                          ex.numeric_derivative(output, px, delta, input_placeholder))
                param_updates.append(update)
            dep_list += dependents

        for ii, ((deriv_s, deriv_n), (_, val)) in enumerate(zip(param_updates, dep_list)):
            ng.testing.assert_allclose(deriv_s(val, input_value),
                                       deriv_n(val, input_value),
                                       rtol=num_rtol,
                                       atol=num_atol)


@pytest.mark.parametrize("sequence_length", [3])
@pytest.mark.parametrize("input_size", [5])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("output_size", [10])
@pytest.mark.parametrize("sum_outputs,concatenate_outputs", [(False, False),
                                                             (True, False),
                                                             (False, True)])
def test_birnn_output_types(recurrent_input, output_size, weight_initializer,
                            sum_outputs, concatenate_outputs):
    """
    Tests that birnns output ops of the right type
    """

    # Generate ngraph RNN
    rnn1 = BiRNN(output_size, init=weight_initializer, activation=Tanh(),
                 reset_cells=True, return_sequence=True,
                 sum_out=sum_outputs, concat_out=concatenate_outputs)
    out = rnn1(recurrent_input)

    if concatenate_outputs:
        assert isinstance(out, ng.ConcatOp), \
            "Output is of type {} instead of {}".format(type(out), ng.ConcatOp)
    elif sum_outputs:
        assert isinstance(out, ng.Add), \
            "Output is of type {} instead of {}".format(type(out), ng.Add)
    else:
        assert isinstance(out, tuple), \
            "Output is of type {} instead of {}".format(type(out), tuple)


@pytest.mark.parametrize("sequence_length", [3])
@pytest.mark.parametrize("input_size", [5])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("output_size", [10])
@pytest.mark.parametrize("sum_outputs,concatenate_outputs", [(False, False),
                                                             (True, False),
                                                             (False, True)])
@pytest.mark.parametrize("extra_feature_axes", [0, 2], indirect=True)
def test_stacked_birnn_construction(recurrent_input, output_size, weight_initializer,
                                    sum_outputs, concatenate_outputs):
    """
    Tests that birnns can be stacked in all of their configurations. If they cannot, an error will
    be thrown, so no assertions are needed.
    """

    # Generate ngraph RNN
    rnn1 = BiRNN(output_size, init=weight_initializer, activation=Tanh(),
                 reset_cells=True, return_sequence=True,
                 sum_out=sum_outputs, concat_out=concatenate_outputs)
    rnn2 = BiRNN(output_size, init=weight_initializer, activation=Tanh(),
                 reset_cells=True, return_sequence=True,
                 sum_out=sum_outputs, concat_out=concatenate_outputs)

    out = rnn1(recurrent_input)
    rnn2(out)


@pytest.config.argon_disabled  # TODO triage
@pytest.mark.flex_disabled
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_length_enc", [5])
@pytest.mark.parametrize("sequence_length_dec", [3])
@pytest.mark.parametrize("input_size", [5])
@pytest.mark.parametrize("hidden_size", [10])
def test_seq2seq_deriv_ref(batch_size, sequence_length_enc, sequence_length_dec, input_size,
                           hidden_size, weight_initializer, bias_initializer,
                           transformer_factory):

    # TODO: are these assumptions true?
    assert batch_size == 1, "the seq2seq reference implementation only support batch size 1"

    # Get input placeholders and numpy arrays
    input_placeholder_enc, input_value_enc, = \
        make_placeholder(input_size, sequence_length_enc, batch_size)
    input_placeholder_dec, input_value_dec, = \
        make_placeholder(input_size, sequence_length_dec, batch_size)

    # Construct encoder weights
    W_in_enc, W_rec_enc, b_enc, _, _ = make_weights(input_placeholder_enc,
                                                    hidden_size,
                                                    weight_initializer,
                                                    bias_initializer,
                                                    init_state=False)

    # Construct decoder weights
    W_in_dec, W_rec_dec, b_dec, _, _ = make_weights(input_placeholder_dec,
                                                    hidden_size,
                                                    weight_initializer,
                                                    bias_initializer,
                                                    init_state=False)

    # Reference numpy seq2seq
    seq2seq_ref = RefSeq2Seq(input_size, hidden_size, decoder_return_sequence=True)
    seq2seq_ref.set_weights(W_in_enc, W_rec_enc, b_enc.reshape(seq2seq_ref.bh_enc.shape),
                            W_in_dec, W_rec_dec, b_dec.reshape(seq2seq_ref.bh_dec.shape))

    # Prepare deltas for gradient check
    output_shape = (hidden_size, sequence_length_dec, batch_size)

    # generate random deltas tensor
    deltas = np.random.randn(*output_shape)

    # the reference code expects these shapes:
    # input_shape: (seq_len, input_size, batch_size)
    # output_shape: (seq_len, hidden_size, batch_size)
    dW_in_enc, dW_rec_enc, db_enc, dW_in_dec, dW_rec_dec, db_dec, encoding_ref, hs_return_dec = \
        seq2seq_ref.lossFun(input_value_enc.transpose([1, 0, 2]),
                            input_value_dec.transpose([1, 0, 2]),
                            deltas.copy().transpose([1, 0, 2]))

    # Generate ngraph Seq2Seq
    rnn_enc_ng = Recurrent(hidden_size, init=W_in_enc, init_inner=W_rec_enc, activation=Tanh(),
                           reset_cells=True, return_sequence=False)
    rnn_dec_ng = Recurrent(hidden_size, init=W_in_dec, init_inner=W_rec_dec, activation=Tanh(),
                           reset_cells=True, return_sequence=True)

    # ngraph fprop graph
    encoding_ng = rnn_enc_ng(input_placeholder_enc, init_state=None)
    output_ng = rnn_dec_ng(input_placeholder_dec, init_state=encoding_ng)

    deltas_constant = ng.constant(deltas, axes=output_ng.axes)
    params = [(rnn_dec_ng.b, db_dec),
              (rnn_dec_ng.W_input, dW_in_dec),
              (rnn_dec_ng.W_recur, dW_rec_dec),
              (rnn_enc_ng.b, db_enc),
              (rnn_enc_ng.W_input, dW_in_enc),
              (rnn_enc_ng.W_recur, dW_rec_enc)]

    with ExecutorFactory() as ex:

        # fprop computations
        fprop_fun = ex.executor([encoding_ng, output_ng],
                                input_placeholder_enc,
                                input_placeholder_dec)

        # gradient computations
        update_funs = []
        for px, _ in params:
            update = ng.deriv(output_ng, px, error=deltas_constant)
            update_funs.append(ex.executor(update, input_placeholder_enc, input_placeholder_dec))

        # check forward pass
        encoding, output = fprop_fun(input_value_enc, input_value_dec)
        ng.testing.assert_allclose(encoding, encoding_ref)
        ng.testing.assert_allclose(np.squeeze(output), np.squeeze(hs_return_dec))

        # check gradient computations
        for update_fun, (_, deriv_ref_val) in zip(update_funs, params):
            grad_neon = update_fun(input_value_enc, input_value_dec)
            ng.testing.assert_allclose(grad_neon,
                                       deriv_ref_val.squeeze(),
                                       rtol=bprop_rtol,
                                       atol=1e-4)
