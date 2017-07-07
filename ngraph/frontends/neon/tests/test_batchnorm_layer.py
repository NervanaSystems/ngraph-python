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

pytestmark = [pytest.mark.transformer_dependent,
              pytest.config.argon_disabled(scope="module")]

rng = RandomTensorGenerator()
rtol = 0
atol = 1e-6
recurrent_atol = 1e-5


class BatchNormReference(object):
    def __init__(self, x,
                 init_gamma=1.0, init_beta=0.0,
                 gmean=0.0, gvar=1.0, rho=0.9, eps=1e-3, axis=(1,)):
        self.red_args = {'axis': axis, 'keepdims': True}
        self.m = np.prod([x.shape[ii] for ii in axis])

        xmean = x.mean(**self.red_args)
        xvar = x.var(**self.red_args)
        self.gamma_scale = init_gamma / np.sqrt(xvar + eps)
        self.xhat = (x - xmean) / np.sqrt(xvar + eps)

        fprop_ref = init_gamma * self.xhat + init_beta
        gmean_ref = xmean.squeeze() * (1.0 - rho) + gmean * rho
        gvar_ref = xvar.squeeze() * (1.0 - rho) + gvar * rho
        self.fprop = (fprop_ref, gmean_ref, gvar_ref)

    def bprop(self, delta):
        dgamma = np.sum(delta * self.xhat, **self.red_args)
        dbeta = np.sum(delta, **self.red_args)
        dx = self.gamma_scale * (delta - (self.xhat * dgamma + dbeta) / self.m)
        return (dx, dgamma.squeeze(), dbeta.squeeze())


class RNNHelper(object):

    def __init__(self, input_placeholder, output_size, RNN, bn_params):

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

        self.rnn_args = dict(nout=output_size,
                             init_inner=self.W_rec,
                             return_sequence=True,
                             activation=Tanh())

        self.reference_rnn = RNN(init=self.W_id, **self.rnn_args)
        self.rnn = RNN(init=self.W_in, batch_norm=True, **self.rnn_args)

        if self.has_gates:
            self.batch_norm_dict = self.rnn.batch_norm
        else:
            self.batch_norm_dict = {'gate': self.rnn.batch_norm}

        self.default_gate = list(self.batch_norm_dict.keys())[0]

        for bn in self.batch_norm_dict.values():
            bn.__dict__.update(bn_params)

    def __getattr__(self, attr):
        if attr in ('gmean', 'gvar', 'gamma', 'beta'):
            return getattr(self.batch_norm_dict[self.default_gate], attr)
        elif attr == 'reference_W_input':
            return self.reference_rnn.W_input[self.default_gate]
        else:
            return super(RNNHelper, self).__getattr__(attr)

    @property
    def has_gates(self):
        return self.reference_rnn.metadata.get('gates') is not None

    # Since we only want to look at the delta back to a single gate, rather than summed over all
    # gates, we can find the dot op between the input and the chosen gate's identity weight matrix
    def get_ancestor_op(self, op):
        for anc_op in ng.Op.ordered_ops([op]):
            if (isinstance(anc_op, ng.DotOp) and
               any(arg.tensor == self.reference_W_input for arg in anc_op.args)):
                return anc_op


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


@pytest.fixture(params=[(1.0, 0.0), (0.6, 0.3)])
def bn_params(request):
    return dict(rho=0.9,
                eps=0.001,
                init_gamma=request.param[0],
                init_beta=request.param[1])


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
        bn_params['gmean'] = 0.0
        bn_params['gvar'] = 1.0

        # Test over 2 iterations to make sure values update properly
        for i in range(2):
            # Generate data
            x = rng.uniform(0, 1, input_placeholder.axes)

            # Compute reference fprop and stats
            batch_norm_reference = BatchNormReference(x, **bn_params)
            out_ref, bn_params['gmean'], bn_params['gvar'] = batch_norm_reference.fprop

            # Compute ngraph fprop and stats
            out = fprop_function(x)
            gm, gv = stats_function()

            assert ng.testing.allclose(out, out_ref, rtol=rtol, atol=atol)
            assert ng.testing.allclose(gm, bn_params['gmean'], rtol=rtol, atol=atol)
            assert ng.testing.allclose(gv, bn_params['gvar'], rtol=rtol, atol=atol)


def test_batchnorm_bprop(input_placeholder, bn_params, transformer_factory):

    layer = BatchNorm(**bn_params)
    fprop = layer(input_placeholder)

    # Derivatives to check
    bprop_vars = [input_placeholder, layer.gamma, layer.beta]

    delta_placeholder = ng.placeholder(fprop.axes)
    bprops = [ng.deriv(fprop, var, delta_placeholder) for var in bprop_vars]

    with ExecutorFactory() as ex:
        # Create derivative executor
        bprop_function = ex.executor(bprops, input_placeholder, delta_placeholder)

        # Generate data
        x = rng.uniform(0, 1, input_placeholder.axes)
        delta = rng.uniform(-.1, .1, delta_placeholder.axes)

        # Compute reference bprop
        dx_ref, dgamma_ref, dbeta_ref = BatchNormReference(x, **bn_params).bprop(delta)

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

    helper = RNNHelper(recurrent_input, output_size, RNN, bn_params)

    # Get batch norm rnn graph
    fprop = helper.rnn(recurrent_input)

    # Get batch norm side effects
    stats = [ng.value_of(helper.gmean), ng.value_of(helper.gvar)]

    # Get reference graph
    reference_fprop = helper.reference_rnn(helper.reference_input)

    with ExecutorFactory() as ex:
        # Compute executors
        fprop_function = ex.executor(fprop, recurrent_input)
        stats_function = ex.executor(stats)
        reference_function = ex.executor(reference_fprop, helper.reference_input)

        # Initial conditions for tracked variables
        bn_params['gmean'] = 0.0
        bn_params['gvar'] = 1.0

        # Need to reduce over two positional axes in reference
        bn_params['axis'] = (1, 2)

        # Test over 2 iterations to make sure values update properly
        for _ in range(2):
            # Get network input values
            input_value = rng.uniform(-1, 1, recurrent_input.axes)

            # Compute reference values
            # First compute the weighted input
            weighted_input = np.dot(helper.W_in, input_value.swapaxes(0, 1))

            # Compute reference batch norm
            batch_norm_reference = BatchNormReference(weighted_input, **bn_params)
            normed_input, bn_params['gmean'], bn_params['gvar'] = batch_norm_reference.fprop

            # Finally, get reference RNN output
            ref = reference_function(normed_input)

            # Get ngraph batch norm RNN output
            out = fprop_function(input_value)
            gmean, gvar = stats_function()

            assert ng.testing.allclose(out, ref, rtol=rtol, atol=recurrent_atol)
            assert ng.testing.allclose(gmean, bn_params['gmean'], rtol=rtol, atol=recurrent_atol)
            assert ng.testing.allclose(gvar, bn_params['gvar'], rtol=rtol, atol=recurrent_atol)


@pytest.mark.parametrize("input_size", [4])
@pytest.mark.parametrize("sequence_length", [2])
@pytest.mark.parametrize("RNN", [Recurrent, LSTM])
def test_recurrent_batchnorm_bprop(RNN, recurrent_input, output_size,
                                   bn_params, transformer_factory):
    """Compare bprop gated RNN with batch norm to numpy batch norm followed by rnn without"""

    helper = RNNHelper(recurrent_input, output_size, RNN, bn_params)

    # Get rnn + batch norm bprop graph
    fprop = helper.rnn(recurrent_input)
    bprop_vars = [recurrent_input, helper.gamma, helper.beta]

    # Get bprop graph
    delta_placeholder = ng.placeholder(fprop.axes)
    bprops = [ng.deriv(fprop, var, delta_placeholder) for var in bprop_vars]

    # Get reference graphs
    reference_fprop = helper.reference_rnn(helper.reference_input)

    # Handle the case where we have gates in the RNN object
    bprop_vars = [helper.reference_input]
    if helper.has_gates:
        bprop_vars.append(helper.get_ancestor_op(reference_fprop))

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

        # Set the reduction axes used for reference
        bn_params['axis'] = (1, 2)

        # Get reference batch normed input
        batch_norm_reference = BatchNormReference(weighted_input, **bn_params)
        normed_input = batch_norm_reference.fprop[0]

        # Reference backprop through RNN
        reference_result = reference_function(normed_input, delta)
        # This is because of a HETR bug where return collections aren't handled properly
        if isinstance(reference_result, tuple):
            rnn_delta = reference_result[0]
        else:
            rnn_delta = reference_result

        # Reference backprop through BN
        dx_ref, dgamma_ref, dbeta_ref = batch_norm_reference.bprop(rnn_delta)

        # Backprop through reference batch norm for a single gate
        if helper.has_gates:
            rnn_gate_delta = reference_result[1]
            _, dgamma_ref, dbeta_ref = batch_norm_reference.bprop(rnn_gate_delta)

        # Backprop through weighted input
        dx_ref = np.dot(helper.W_in.T, dx_ref.swapaxes(0, 1))

        # Compute ngraph bprop
        dx, dgamma, dbeta = bprop_function(input_value, delta)

        assert ng.testing.allclose(dx, dx_ref, rtol=rtol, atol=recurrent_atol)
        assert ng.testing.allclose(dgamma, dgamma_ref, rtol=rtol, atol=recurrent_atol)
        assert ng.testing.allclose(dbeta, dbeta_ref, rtol=rtol, atol=recurrent_atol)
