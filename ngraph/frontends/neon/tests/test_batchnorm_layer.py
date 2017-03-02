# ----------------------------------------------------------------------------
# Copyright 2015-2017 Nervana Systems Inc.
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
Test of the batchnorm layer
'''
import pytest
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import BatchNorm, Recurrent, LSTM, Tanh, GlorotInit
from ngraph.testing.random import RandomTensorGenerator
from ngraph.testing.execution import ExecutorFactory

rng = RandomTensorGenerator()
rtol = 0
atol = 1e-6
recurrent_atol = 1e-5


def batch_norm_reference_fprop(x, gmean_ref=0.0, gvar_ref=1.0, rho=0.9, epsilon=1e-3,
                               gamma=1, beta=0, axis=1):

    xmean = x.mean(axis=axis, keepdims=True)
    xvar = x.var(axis=axis, keepdims=True)
    out_ref = gamma * (x - xmean) / np.sqrt(xvar + epsilon) + beta
    gmean_ref = xmean.squeeze() * (1.0 - rho) + gmean_ref * rho
    gvar_ref = xvar.squeeze() * (1.0 - rho) + gvar_ref * rho

    return out_ref, gmean_ref, gvar_ref


def batch_norm_reference_bprop(delta, x, gamma=1, epsilon=1e-3, axis=1):

    # Compute x_hat = (x - mu) / std
    xmean = x.mean(axis=axis, keepdims=True)
    xvar = x.var(axis=axis, keepdims=True)
    xhat = (x - xmean) / np.sqrt(xvar + epsilon)

    # Get overall size of reduction axes
    m = x.shape[axis] if not isinstance(axis, tuple) else np.float32(np.prod([x.shape[ii] for ii in axis]))

    # Compute derivatives
    dgamma = np.sum(delta * xhat, axis=axis, keepdims=True)
    dbeta = np.sum(delta, axis=axis, keepdims=True)
    dx = gamma / np.sqrt(xvar + epsilon) * (delta - (xhat * dgamma + dbeta) / m)  # Possibly gamma only on delta

    return dx, dgamma.squeeze(), dbeta.squeeze()


@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("input_size", [4, 10, 32])
@pytest.mark.parametrize("rho,epsilon", [(0.2, 0.1)])
@pytest.mark.parametrize("beta,gamma", [(0, 1), (.1, 1.2)])
def test_batchnorm_fprop(batch_size, input_size, rho, epsilon, beta, gamma, transformer_factory):
    """This checks that that we are doing batch norm across a feature make_axis
    and properly tracking the side effect variables
    """

    # set inputs
    N = ng.make_axis(batch_size, name='N')
    F = ng.make_axis(input_size)

    input_placeholder = ng.placeholder([F, N])
    layer = BatchNorm(rho=rho, eps=epsilon, init_gamma=gamma, init_beta=beta)
    fprop = layer(input_placeholder)

    with ExecutorFactory() as ex:
        fprop_function = ex.transformer.computation(fprop, input_placeholder)
        stats_function = ex.transformer.computation([ng.value_of(layer.gmean),
                                                     ng.value_of(layer.gvar)])

        # initial conditions for tracked variables
        gmean_ref, gvar_ref = 0.0, 1.0

        # create data
        for i in range(2):
            x = np.random.random((input_size, batch_size)).astype(np.float32)

            out = fprop_function(x)
            gm, gv = stats_function()

            out_ref, gmean_ref, gvar_ref = batch_norm_reference_fprop(x, gmean_ref, gvar_ref,
                                                                      rho, epsilon, gamma, beta)

            assert ng.testing.allclose(out, out_ref, rtol=rtol, atol=atol)
            assert ng.testing.allclose(gm, gmean_ref, rtol=rtol, atol=atol)
            assert ng.testing.allclose(gv, gvar_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("epsilon", [.1])
@pytest.mark.parametrize("beta,gamma", [(0, 1), (.5, .6)])
def test_batchnorm_bprop(batch_size, input_size, epsilon, beta, gamma, transformer_factory):

    # set inputs
    N = ng.make_axis(batch_size, name='N')
    F = ng.make_axis(input_size)
    axes = ng.make_axes([F, N])

    input_placeholder = ng.placeholder(axes)
    delta_placeholder = ng.placeholder(axes)
    layer = BatchNorm(eps=epsilon, init_gamma=gamma, init_beta=beta)
    fprop = layer(input_placeholder)
    bprop_vars = [input_placeholder,
                  layer.gamma,
                  layer.beta]
    bprops = [ng.deriv(fprop, var, delta_placeholder) for var in bprop_vars]

    with ExecutorFactory() as ex:
        bprop_function = ex.executor(bprops, input_placeholder, delta_placeholder)
        x = rng.uniform(0, 1, axes)
        delta = rng.uniform(0, 1, axes)

        dx_ref, dgamma_ref, dbeta_ref = batch_norm_reference_bprop(delta, x, gamma=gamma,
                                                                   epsilon=epsilon)
        dx, dgamma, dbeta = bprop_function(x, delta)

        assert ng.testing.allclose(dx, dx_ref, rtol=rtol, atol=atol)
        assert ng.testing.allclose(dgamma, dgamma_ref, rtol=rtol, atol=atol)
        assert ng.testing.allclose(dbeta, dbeta_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("input_size", [4])
@pytest.mark.parametrize("hidden_size", [10])
@pytest.mark.parametrize("sequence_length", [5])
@pytest.mark.parametrize("RNN", [Recurrent, LSTM])
def test_recurrent_batchnorm_fprop(RNN, batch_size, input_size, hidden_size, sequence_length,
                                   transformer_factory):
    """Compare RNN with batch norm to rnn without using normalized weighted inputs. """

    # Set up axes
    F = ng.make_axis(length=input_size, name="input")
    T = ng.make_axis(length=sequence_length, name="R")
    N = ng.make_axis(length=batch_size, name="N")
    H = ng.make_axis(length=hidden_size, name="hidden")
    F2 = ng.make_axis(length=hidden_size, name="weighted_input")

    # Make input placeholder
    input_axes = ng.make_axes([F, T, N])
    input_placeholder = ng.placeholder(axes=input_axes)
    normed_input_placeholder = ng.placeholder(axes=[F2, T, N])

    # Create weight matrices
    w_rec_axes = ng.make_axes([H, H - 1])
    w_in_axes = ng.make_axes([H, F - 1])
    hidden_weights = rng.uniform(-1, 1, w_rec_axes)
    input_weights = rng.uniform(-1, 1, w_in_axes)
    identity_weights = np.eye(hidden_size).astype("float32")

    # Generate an RNN with batch norm turned on
    batch_norm = BatchNorm()
    rnn = RNN(hidden_size, init=input_weights, init_inner=hidden_weights,
              return_sequence=True, batch_norm=batch_norm, activation=Tanh())

    # Generate an RNN with no batch norm
    reference_rnn = RNN(hidden_size, init=identity_weights, init_inner=hidden_weights,
                        return_sequence=True, activation=Tanh())

    # Get batch norm rnn graph
    fprop = rnn(input_placeholder)

    # Get batch norm side effects
    if isinstance(rnn.batch_norm, dict):  # e.g. LSTM
        # rnn has multiple gates, so just look at one of them.
        k = rnn.batch_norm.keys()[0]
        stats = [ng.value_of(rnn.batch_norm[k].gmean),
                 ng.value_of(rnn.batch_norm[k].gvar)]
    else:
        stats = [ng.value_of(rnn.batch_norm.gmean),
                 ng.value_of(rnn.batch_norm.gvar)]

    # Get reference graph
    reference_fprop = reference_rnn(normed_input_placeholder)

    # Begin execution
    with ExecutorFactory() as ex:
        fprop_function = ex.executor(fprop, input_placeholder)
        stats_function = ex.executor(stats)
        reference_function = ex.executor(reference_fprop, normed_input_placeholder)

        gmean_ref, gvar_ref = 0.0, 1.0
        for _ in range(2):
            # Get network input values
            input_value = rng.uniform(0, 1, input_axes)
            weighted_input = np.dot(input_weights, input_value.transpose([1, 0, 2]))

            # Get reference batch normed input
            normed_input, gmean_ref, gvar_ref = batch_norm_reference_fprop(weighted_input, gmean_ref,
                                                                           gvar_ref, axis=(1, 2))

            # Get reference RNN output
            ref = reference_function(normed_input)

            # Get batch norm RNN output
            out = fprop_function(input_value)
            gmean, gvar = stats_function()

            assert ng.testing.allclose(out, ref, rtol=rtol, atol=atol)
            assert ng.testing.allclose(gmean, gmean_ref, rtol=rtol, atol=atol)
            assert ng.testing.allclose(gvar, gvar_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("input_size", [4])
@pytest.mark.parametrize("hidden_size", [10])
@pytest.mark.parametrize("sequence_length", [5])
@pytest.mark.parametrize("RNN", [Recurrent])
@pytest.mark.parametrize("beta,gamma", [(0, 1), (.5, .6)])
def test_recurrent_batchnorm_bprop(batch_size, input_size, hidden_size, sequence_length, RNN, beta,
                                   gamma):

    # Set up axes
    F = ng.make_axis(length=input_size, name="input")
    T = ng.make_axis(length=sequence_length, name="R")
    N = ng.make_axis(length=batch_size, name="N")
    H = ng.make_axis(length=hidden_size, name="hidden")
    F2 = ng.make_axis(length=hidden_size, name="weighted_input")

    # Make input placeholders
    input_axes = ng.make_axes([F, T, N])
    input_placeholder = ng.placeholder(axes=input_axes)
    normed_input_placeholder = ng.placeholder(axes=[F2, T, N])

    # Create weight matrices
    w_rec_axes = ng.make_axes([H, H - 1])
    w_in_axes = ng.make_axes([H, F - 1])
    hidden_weights = GlorotInit()(w_rec_axes)
    input_weights = GlorotInit()(w_in_axes)
    identity_weights = np.eye(hidden_size).astype("float32")

    # Generate an RNN with batch norm turned on
    batch_norm = BatchNorm(init_gamma=gamma, init_beta=beta)
    rnn = RNN(hidden_size, init=input_weights, init_inner=hidden_weights,
              return_sequence=True, batch_norm=batch_norm, activation=Tanh())

    # Generate an RNN with no batch norm
    reference_rnn = RNN(hidden_size, init=identity_weights, init_inner=hidden_weights,
                        return_sequence=True, activation=Tanh())

    # Get rnn + batch norm graph
    fprop = rnn(input_placeholder)
    bprop_vars = [input_placeholder]
    if isinstance(rnn.batch_norm, dict):  # e.g. LSTM
        # rnn has multiple gates, so just look at one of them.
        k = rnn.batch_norm.keys()[0]
        bprop_vars += [rnn.batch_norm[k].gamma,
                       rnn.batch_norm[k].beta]
    else:
        bprop_vars += [rnn.batch_norm.gamma,
                       rnn.batch_norm.beta]

    # Get bprop graph
    delta_placeholder = ng.placeholder(fprop.axes)
    bprops = [ng.deriv(fprop, var, delta_placeholder) for var in bprop_vars]

    # Get reference graphs
    reference_fprop = reference_rnn(normed_input_placeholder)
    reference_delta_placeholder = ng.placeholder(reference_fprop.axes)
    reference_bprop = ng.deriv(reference_fprop, normed_input_placeholder,
                               reference_delta_placeholder)

    # Begin execution
    with ExecutorFactory() as ex:
        bprop_function = ex.executor(bprops, input_placeholder, delta_placeholder)
        reference_function = ex.executor(reference_bprop, normed_input_placeholder,
                                         reference_delta_placeholder)

        input_value = rng.uniform(0, 1, input_axes)
        delta = rng.uniform(-1, 1, fprop.axes)
        dx, dgamma, dbeta = bprop_function(input_value, delta)

        # Compute reference weighted input
        weighted_input = np.dot(input_weights, input_value.swapaxes(0, 1))

        # Get reference batch normed input
        normed_input = batch_norm_reference_fprop(weighted_input, gamma=gamma, beta=beta,
                                                  axis=(1, 2))[0]

        # Reference backprop through RNN
        rnn_delta = reference_function(normed_input, delta)

        # Reference backprop through BN
        dx_ref, dgamma_ref, dbeta_ref = batch_norm_reference_bprop(rnn_delta, weighted_input,
                                                                   gamma=gamma, axis=(1, 2))

        # Backprop through weighted input
        dx_ref = np.dot(input_weights.T, dx_ref.swapaxes(0, 1))

        assert ng.testing.allclose(dx, dx_ref, rtol=rtol, atol=recurrent_atol)
        assert ng.testing.allclose(dgamma, dgamma_ref, rtol=rtol, atol=recurrent_atol)
        assert ng.testing.allclose(dbeta, dbeta_ref, rtol=rtol, atol=recurrent_atol)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("input_size", [4])
@pytest.mark.parametrize("hidden_size", [10])
@pytest.mark.parametrize("sequence_length", [5])
@pytest.mark.parametrize("RNN", [LSTM])
@pytest.mark.parametrize("beta,gamma", [(0, 1), (.5, .6)])
def test_gated_recurrent_batchnorm_bprop(batch_size, input_size, hidden_size, sequence_length, RNN,
                                         beta, gamma):

    # Set up axes
    F = ng.make_axis(length=input_size, name="input")
    T = ng.make_axis(length=sequence_length, name="R")
    N = ng.make_axis(length=batch_size, name="N")
    H = ng.make_axis(length=hidden_size, name="hidden")
    F2 = ng.make_axis(length=hidden_size, name="weighted_input")

    # Make input placeholders
    input_axes = ng.make_axes([F, T, N])
    input_placeholder = ng.placeholder(axes=input_axes)
    normed_input_placeholder = ng.placeholder(axes=[F2, T, N])

    # Create weight matrices
    w_rec_axes = ng.make_axes([H, H - 1])
    w_in_axes = ng.make_axes([H, F - 1])
    hidden_weights = GlorotInit()(w_rec_axes)
    input_weights = GlorotInit()(w_in_axes)
    identity_weights = np.eye(hidden_size).astype("float32")

    # Generate an RNN with batch norm turned on
    batch_norm = BatchNorm(init_gamma=gamma, init_beta=beta)
    rnn = RNN(hidden_size, init=input_weights, init_inner=hidden_weights,
              return_sequence=True, batch_norm=batch_norm, activation=Tanh())

    # Generate an RNN with no batch norm
    reference_rnn = RNN(hidden_size, init=identity_weights, init_inner=hidden_weights,
                        return_sequence=True, activation=Tanh())

    # Get rnn + batch norm graph
    fprop = rnn(input_placeholder)
    # rnn has multiple gates, so just look at one of them.
    k = rnn.batch_norm.keys()[0]
    bprop_vars = [input_placeholder,
                  rnn.batch_norm[k].gamma,
                  rnn.batch_norm[k].beta]

    # Get bprop graph
    delta_placeholder = ng.placeholder(fprop.axes)
    bprops = [ng.deriv(fprop, var, delta_placeholder) for var in bprop_vars]

    # Get reference graphs


    def filter_ancestors(op, func):
        filtered_ops = list()
        for anc_op in ng.Op.ordered_ops([op]):
            if func(anc_op) is True:
                filtered_ops.append(anc_op)

        return filtered_ops


    filter_func = lambda op: (isinstance(op, ng.DotOp) and
                              any(arg.tensor == reference_rnn.W_input[k] for arg in op.args))
    reference_fprop = reference_rnn(normed_input_placeholder)
    bprop_vars = [normed_input_placeholder,
                  filter_ancestors(reference_fprop, filter_func)[0]]
    reference_delta_placeholder = ng.placeholder(reference_fprop.axes)
    reference_bprop = [ng.deriv(reference_fprop, var,
                                reference_delta_placeholder) for var in bprop_vars]

    # Begin execution
    with ExecutorFactory() as ex:
        bprop_function = ex.executor(bprops, input_placeholder, delta_placeholder)
        reference_function = ex.executor(reference_bprop, normed_input_placeholder,
                                         reference_delta_placeholder)

        input_value = rng.uniform(0, 1, input_axes)
        delta = rng.uniform(-1, 1, fprop.axes)
        dx, dgamma, dbeta = bprop_function(input_value, delta)

        # Compute reference weighted input
        weighted_input = np.dot(input_weights, input_value.swapaxes(0, 1))

        # Get reference batch normed input
        normed_input = batch_norm_reference_fprop(weighted_input, gamma=gamma, beta=beta,
                                                  axis=(1, 2))[0]

        # Reference backprop through RNN
        rnn_delta, rnn_gate_delta = reference_function(normed_input, delta)

        # Reference backprop through BN
        dx_ref = batch_norm_reference_bprop(rnn_delta, weighted_input,
                                            gamma=gamma, axis=(1, 2))[0]

        _, dgamma_ref, dbeta_ref = batch_norm_reference_bprop(rnn_gate_delta, weighted_input,
                                                              gamma=gamma, axis=(1, 2))

        # Backprop through weighted input
        dx_ref = np.dot(input_weights.T, dx_ref.swapaxes(0, 1))

        assert ng.testing.allclose(dx, dx_ref, rtol=rtol, atol=recurrent_atol)
        assert ng.testing.allclose(dgamma, dgamma_ref, rtol=rtol, atol=recurrent_atol)
        assert ng.testing.allclose(dbeta, dbeta_ref, rtol=rtol, atol=recurrent_atol)
