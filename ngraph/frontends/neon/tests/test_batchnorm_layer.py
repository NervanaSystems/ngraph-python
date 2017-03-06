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
from ngraph.frontends.neon import BatchNorm, Recurrent, LSTM, Tanh
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
    if not isinstance(axis, tuple):
        m = x.shape[axis]
    else:
        m = np.prod([x.shape[ii] for ii in axis])

    # Compute derivatives
    dgamma = np.sum(delta * xhat, axis=axis, keepdims=True)
    dbeta = np.sum(delta, axis=axis, keepdims=True)
    dx = gamma / np.sqrt(xvar + epsilon) * (delta - (xhat * dgamma + dbeta) / m)

    return dx, dgamma.squeeze(), dbeta.squeeze()


class RNNHelper(object):

    def __init__(self, input_placeholder, output_size):

        # Set up axes
        F, T, N = tuple(input_placeholder.axes)
        H = ng.make_axis(length=output_size, name="hidden")
        H2 = ng.make_axis(length=output_size, name="hidden_tmp")

        self.input_placeholder = input_placeholder

        # Make reference placeholder
        self.reference_input = ng.placeholder(axes=[H, T, N])

        # Create weight matrices
        w_rec_axes = ng.make_axes([H, H2])
        w_in_axes = ng.make_axes([H, F])
        self.W_rec = rng.uniform(-1, 1, w_rec_axes)
        self.W_in = rng.uniform(-1, 1, w_in_axes)
        self.W_id = np.eye(output_size).astype("float32")

    @staticmethod
    def set_batch_norm_params(rnn, **kwargs):
        for key, value in kwargs.items():
            if isinstance(rnn.batch_norm, dict):
                for bn in rnn.batch_norm.values():
                    setattr(bn, key, value)
            else:
                setattr(rnn.batch_norm, key, value)

    @staticmethod
    def get_batch_norm_params(rnn):
        if isinstance(rnn.batch_norm, dict):  # e.g. LSTM
            # rnn has multiple gates, so just look at one of them.
            k = list(rnn.batch_norm.keys())[0]
            bn = rnn.batch_norm[k]
        else:
            bn = rnn.batch_norm

        return bn.gmean, bn.gvar


# TODO: Move the following *_size fixtures to conftest.py and refactor other tests to use them
@pytest.fixture(params=[32])
def batch_size(request):
    return request.param


@pytest.fixture(params=[5])
def sequence_length(request):
    return request.param


@pytest.fixture(params=[4, 32])
def input_size(request):
    return request.param


@pytest.fixture(params=[10])
def output_size(request):
    return request.param


@pytest.fixture(scope='module', params=[(.2, .1, 1, 0),
                                        (.2, .1, .6, .3)])
def bn_params(request):
    return dict(zip(["rho", "eps", "init_gamma", "init_beta"],
                    request.param))


def test_batchnorm_fprop(input_placeholder, bn_params, transformer_factory):
    """This checks that that we are doing batch norm across a feature make_axis
    and properly tracking the side effect variables
    """

    layer = BatchNorm(**bn_params)
    fprop = layer(input_placeholder)

    with ExecutorFactory() as ex:
        # Compute executors
        fprop_function = ex.executor(fprop, input_placeholder)
        stats_function = ex.executor([ng.value_of(layer.gmean),
                                      ng.value_of(layer.gvar)])

        # Initial conditions for tracked variables
        gmean_ref, gvar_ref = 0.0, 1.0

        # Test over 2 iterations to make sure values update properly
        for i in range(2):
            # Generate data
            x = rng.uniform(0, 1, input_placeholder.axes)

            # Compute reference fprop and stats
            out_ref, gmean_ref, gvar_ref = batch_norm_reference_fprop(x, gmean_ref, gvar_ref,
                                                                      bn_params["rho"],
                                                                      bn_params["eps"],
                                                                      bn_params["init_gamma"],
                                                                      bn_params["init_beta"])

            # Compute ngraph fprop and stats
            out = fprop_function(x)
            gm, gv = stats_function()

            assert ng.testing.allclose(out, out_ref, rtol=rtol, atol=atol)
            assert ng.testing.allclose(gm, gmean_ref, rtol=rtol, atol=atol)
            assert ng.testing.allclose(gv, gvar_ref, rtol=rtol, atol=atol)


def test_batchnorm_bprop(input_placeholder, bn_params, transformer_factory):

    layer = BatchNorm(**bn_params)
    fprop = layer(input_placeholder)

    # Derivatives to check
    bprop_vars = [input_placeholder,
                  layer.gamma,
                  layer.beta]
    delta_placeholder = ng.placeholder(fprop.axes)
    bprops = [ng.deriv(fprop, var, delta_placeholder) for var in bprop_vars]

    with ExecutorFactory() as ex:
        # Create derivative executor
        bprop_function = ex.executor(bprops, input_placeholder, delta_placeholder)

        # Generate data
        x = rng.uniform(0, 1, input_placeholder.axes)
        delta = rng.uniform(-.1, .1, delta_placeholder.axes)

        # Compute reference bprop
        dx_ref, dgamma_ref, dbeta_ref = batch_norm_reference_bprop(delta, x,
                                                                   gamma=bn_params["init_gamma"],
                                                                   epsilon=bn_params["eps"])
        # Compute ngraph bprop
        dx, dgamma, dbeta = bprop_function(x, delta)

        assert ng.testing.allclose(dx, dx_ref, rtol=rtol, atol=atol)
        assert ng.testing.allclose(dgamma, dgamma_ref, rtol=rtol, atol=atol)
        assert ng.testing.allclose(dbeta, dbeta_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("input_size", [4])
@pytest.mark.parametrize("sequence_length", [2])
@pytest.mark.parametrize("RNN", [Recurrent, LSTM])
def test_recurrent_batchnorm_fprop(RNN, recurrent_input, output_size,
                                   bn_params, transformer_factory):
    """Compare fprop RNN with batch norm to numpy batch norm followed by rnn without"""

    helper = RNNHelper(recurrent_input, output_size)

    rnn = RNN(output_size, init=helper.W_in, init_inner=helper.W_rec,
              batch_norm=True, return_sequence=True, activation=Tanh())
    reference_rnn = RNN(output_size, init=helper.W_id, init_inner=helper.W_rec,
                        return_sequence=True, activation=Tanh())

    # Set batch norm parameters
    helper.set_batch_norm_params(rnn, **bn_params)

    # Get batch norm rnn graph
    fprop = rnn(recurrent_input)

    # Get batch norm side effects
    gmean, gvar = helper.get_batch_norm_params(rnn)
    stats = [ng.value_of(gmean), ng.value_of(gvar)]

    # Get reference graph
    normed_recurrent_input = helper.reference_input
    reference_fprop = reference_rnn(normed_recurrent_input)

    with ExecutorFactory() as ex:
        # Compute executors
        fprop_function = ex.executor(fprop, recurrent_input)
        stats_function = ex.executor(stats)
        reference_function = ex.executor(reference_fprop, normed_recurrent_input)

        # Initial conditions for tracked variables
        gmean_ref, gvar_ref = 0.0, 1.0

        # Test over 2 iterations to make sure values update properly
        for _ in range(2):
            # Get network input values
            input_value = rng.uniform(-1, 1, recurrent_input.axes)

            # Compute reference values
            # First compute the weighted input
            weighted_input = np.dot(helper.W_in, input_value.swapaxes(0, 1))

            # Compute reference batch norm
            (normed_input,
             gmean_ref,
             gvar_ref) = batch_norm_reference_fprop(weighted_input,
                                                    gmean_ref,
                                                    gvar_ref,
                                                    bn_params["rho"],
                                                    bn_params["eps"],
                                                    bn_params["init_gamma"],
                                                    bn_params["init_beta"],
                                                    axis=(1, 2))

            # Finally, get reference RNN output
            ref = reference_function(normed_input)

            # Get ngraph batch norm RNN output
            out = fprop_function(input_value)
            gmean, gvar = stats_function()

            assert ng.testing.allclose(out, ref, rtol=rtol, atol=recurrent_atol)
            assert ng.testing.allclose(gmean, gmean_ref, rtol=rtol, atol=recurrent_atol)
            assert ng.testing.allclose(gvar, gvar_ref, rtol=rtol, atol=recurrent_atol)


@pytest.mark.parametrize("input_size", [4])
@pytest.mark.parametrize("sequence_length", [2])
@pytest.mark.parametrize("RNN", [Recurrent])
def test_recurrent_batchnorm_bprop(RNN, recurrent_input, output_size,
                                   bn_params, transformer_factory):
    """Compare bprop RNN with batch norm to numpy batch norm followed by rnn without"""

    helper = RNNHelper(recurrent_input, output_size)

    # Generate an RNN with batch norm turned on
    rnn = RNN(output_size, init=helper.W_in, init_inner=helper.W_rec,
              return_sequence=True, batch_norm=True, activation=Tanh())

    # Set batch norm params
    helper.set_batch_norm_params(rnn, **bn_params)

    # Generate an RNN with no batch norm
    reference_rnn = RNN(output_size, init=helper.W_id, init_inner=helper.W_rec,
                        return_sequence=True, activation=Tanh())

    # Get rnn + batch norm bprop graph
    fprop = rnn(recurrent_input)
    bprop_vars = [recurrent_input,
                  rnn.batch_norm.gamma,
                  rnn.batch_norm.beta]
    delta_placeholder = ng.placeholder(fprop.axes)
    bprops = [ng.deriv(fprop, var, delta_placeholder) for var in bprop_vars]

    # Get reference bprop graph
    reference_fprop = reference_rnn(helper.reference_input)
    reference_delta_placeholder = ng.placeholder(reference_fprop.axes)
    reference_bprop = ng.deriv(reference_fprop, helper.reference_input,
                               reference_delta_placeholder)

    # Begin execution
    with ExecutorFactory() as ex:
        bprop_function = ex.executor(bprops, recurrent_input, delta_placeholder)
        reference_function = ex.executor(reference_bprop, helper.reference_input,
                                         reference_delta_placeholder)

        input_value = rng.uniform(0, 1, recurrent_input.axes)
        delta = rng.uniform(-.1, .1, fprop.axes)

        # Compute reference bprop
        # Compute reference weighted input
        weighted_input = np.dot(helper.W_in, input_value.swapaxes(0, 1))

        # Get reference batch normed input
        normed_input = batch_norm_reference_fprop(weighted_input,
                                                  0.0,
                                                  1.0,
                                                  bn_params["rho"],
                                                  bn_params["eps"],
                                                  bn_params["init_gamma"],
                                                  bn_params["init_beta"],
                                                  axis=(1, 2))[0]

        # Reference backprop through RNN
        rnn_delta = reference_function(normed_input, delta)

        # Backprop through reference BN
        dx_ref, dgamma_ref, dbeta_ref = batch_norm_reference_bprop(rnn_delta, weighted_input,
                                                                   epsilon=bn_params["eps"],
                                                                   gamma=bn_params["init_gamma"],
                                                                   axis=(1, 2))

        # Backprop through weighted input
        dx_ref = np.dot(helper.W_in.T, dx_ref.swapaxes(0, 1))

        # Compute ngraph bprop
        dx, dgamma, dbeta = bprop_function(input_value, delta)

        assert ng.testing.allclose(dx, dx_ref, rtol=rtol, atol=recurrent_atol)
        assert ng.testing.allclose(dgamma, dgamma_ref, rtol=rtol, atol=recurrent_atol)
        assert ng.testing.allclose(dbeta, dbeta_ref, rtol=rtol, atol=recurrent_atol)


@pytest.mark.parametrize("input_size", [4])
@pytest.mark.parametrize("sequence_length", [2])
@pytest.mark.parametrize("RNN", [LSTM])
def test_gated_recurrent_batchnorm_bprop(RNN, recurrent_input, output_size, bn_params,
                                         transformer_factory):
    """Compare bprop gated RNN with batch norm to numpy batch norm followed by rnn without"""

    helper = RNNHelper(recurrent_input, output_size)

    # Generate an RNN with batch norm turned on
    rnn = RNN(output_size, init=helper.W_in, init_inner=helper.W_rec,
              return_sequence=True, batch_norm=True, activation=Tanh())

    # Set batch norm params
    helper.set_batch_norm_params(rnn, **bn_params)

    # Generate an RNN with no batch norm
    reference_rnn = RNN(output_size, init=helper.W_id, init_inner=helper.W_rec,
                        return_sequence=True, activation=Tanh())

    # Get rnn + batch norm graph
    fprop = rnn(recurrent_input)
    # rnn has multiple gates, so just look at one of them.
    k = list(rnn.batch_norm.keys())[0]
    bprop_vars = [recurrent_input,
                  rnn.batch_norm[k].gamma,
                  rnn.batch_norm[k].beta]

    # Get bprop graph
    delta_placeholder = ng.placeholder(fprop.axes)
    bprops = [ng.deriv(fprop, var, delta_placeholder) for var in bprop_vars]

    # Get reference graphs
    # Since we only want to look at the delta back to a single gate, rather than summed over all
    # gates, we can find the dot op between the input and the chosen gate's identity weight matrix
    def filter_ancestors(op, func):
        """ Filter ancestors of an op according to func"""
        filtered_ops = list()
        for anc_op in ng.Op.ordered_ops([op]):
            if func(anc_op) is True:
                filtered_ops.append(anc_op)

        return filtered_ops

    filter_func = lambda op: (isinstance(op, ng.DotOp) and
                              any(arg.tensor == reference_rnn.W_input[k] for arg in op.args))

    reference_fprop = reference_rnn(helper.reference_input)
    bprop_vars = [helper.reference_input,
                  filter_ancestors(reference_fprop, filter_func)[0]]
    reference_delta_placeholder = ng.placeholder(reference_fprop.axes)
    reference_bprop = [ng.deriv(reference_fprop, var,
                                reference_delta_placeholder) for var in bprop_vars]

    # Begin execution
    with ExecutorFactory() as ex:
        bprop_function = ex.executor(bprops, recurrent_input, delta_placeholder)
        reference_function = ex.executor(reference_bprop, helper.reference_input,
                                         reference_delta_placeholder)

        # Create data
        input_value = rng.uniform(0, 1, recurrent_input.axes)
        delta = rng.uniform(-.1, .1, fprop.axes)

        # Compute reference weighted input
        weighted_input = np.dot(helper.W_in, input_value.swapaxes(0, 1))

        # Get reference batch normed input
        normed_input = batch_norm_reference_fprop(weighted_input,
                                                  0.0,
                                                  1.0,
                                                  bn_params["rho"],
                                                  bn_params["eps"],
                                                  bn_params["init_gamma"],
                                                  bn_params["init_beta"],
                                                  axis=(1, 2))[0]

        # Reference backprop through RNN
        rnn_delta, rnn_gate_delta = reference_function(normed_input, delta)

        # Reference backprop through BN
        dx_ref = batch_norm_reference_bprop(rnn_delta, weighted_input,
                                            epsilon=bn_params["eps"],
                                            gamma=bn_params["init_gamma"],
                                            axis=(1, 2))[0]

        # Backprop through reference batch norm for a single gate
        _, dgamma_ref, dbeta_ref = batch_norm_reference_bprop(rnn_gate_delta, weighted_input,
                                                              epsilon=bn_params["eps"],
                                                              gamma=bn_params["init_gamma"],
                                                              axis=(1, 2))

        # Backprop through weighted input
        dx_ref = np.dot(helper.W_in.T, dx_ref.swapaxes(0, 1))

        # Compute ngraph bprop
        dx, dgamma, dbeta = bprop_function(input_value, delta)

        assert ng.testing.allclose(dx, dx_ref, rtol=rtol, atol=recurrent_atol)
        assert ng.testing.allclose(dgamma, dgamma_ref, rtol=rtol, atol=recurrent_atol)
        assert ng.testing.allclose(dbeta, dbeta_ref, rtol=rtol, atol=recurrent_atol)
