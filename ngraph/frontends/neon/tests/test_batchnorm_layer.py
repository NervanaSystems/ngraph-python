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

def batch_norm_reference(x, gmean_ref=0.0, gvar_ref=1.0, rho=0.9, epsilon=1e-3, axis=1):

    xmean = x.mean(axis=axis, keepdims=True)
    xvar = x.var(axis=axis, keepdims=True)
    out_ref = (x - xmean) / np.sqrt(xvar + epsilon)
    gmean_ref = xmean.squeeze() * (1.0 - rho) + gmean_ref * rho
    gvar_ref = xvar.squeeze() * (1.0 - rho) + gvar_ref * rho

    return out_ref, gmean_ref, gvar_ref



@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("input_size", [4, 10, 32])
@pytest.mark.parametrize("rho,epsilon", [(0.2, 0.1)])
def test_batchnorm_fprop(batch_size, input_size, rho, epsilon, transformer_factory):
    # This checks that that we are doing batch norm across a feature make_axis
    # and properly tracking the side effect variables
    np.random.seed(0)
    # set inputs
    N = ng.make_axis(batch_size, name='N')
    F = ng.make_axis(input_size)

    input_placeholder = ng.placeholder([F, N])
    layer = BatchNorm(rho, epsilon)
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

            out_ref, gmean_ref, gvar_ref = batch_norm_reference(x, gmean_ref, gvar_ref,
                                                                rho, epsilon)

            assert ng.testing.allclose(out,
                                       out_ref, atol=1e-6), '%e' % np.max(np.abs(out - out_ref))
            assert ng.testing.allclose(gm,
                                       gmean_ref, atol=1e-6), '%e' % np.max(np.abs(gm - gmean_ref))
            assert ng.testing.allclose(gv,
                                       gvar_ref, atol=1e-6), '%e' % np.max(np.abs(gv - gvar_ref))


# @pytest.mark.skip
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
    T = ng.make_axis(length=sequence_length, recurrent=True)
    N = ng.make_axis(length=batch_size, batch=True)
    H = ng.make_axis(length=hidden_size, name="hidden")
    F2 = ng.make_axis(length=hidden_size, name="weighted_hidden")

    # Make input placeholder
    input_axes = ng.make_axes([F, T, N])
    input_placeholder = ng.placeholder(axes=input_axes)
    normed_input_placeholder = ng.placeholder(axes=[F2, T, N])

    # Create weight matrices
    w_rec_axes = ng.make_axes([H, H - 1])
    w_in_axes = ng.make_axes([H, F - 1])
    hidden_weights = rng.uniform(-1, 1, w_rec_axes)
    input_weights = rng.uniform(-1, 1, w_in_axes)
    identity_weights = np.eye(hidden_size)

    # Generate an RNN with batch norm turned on
    batch_norm = BatchNorm()
    rnn = RNN(hidden_size, init=input_weights, init_inner=hidden_weights,
              return_sequence=True, batch_norm=batch_norm, activation=Tanh())
    reference_rnn = RNN(hidden_size, init=identity_weights, init_inner=hidden_weights,
                        return_sequence=True, activation=Tanh())

    # Get output placeholders
    fprop = rnn(input_placeholder)
    if isinstance(rnn.batch_norm, dict):
        # rnn has multiple gates, so just look at one of them.
        k = rnn.batch_norm.keys()[0]
        stats = [ng.value_of(rnn.batch_norm[k].gmean),
                 ng.value_of(rnn.batch_norm[k].gvar)]
    else:
        stats = [ng.value_of(rnn.batch_norm.gmean),
                 ng.value_of(rnn.batch_norm.gvar)]
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
            normed_input, gmean_ref, gvar_ref = batch_norm_reference(weighted_input, gmean_ref,
                                                                     gvar_ref, axis=(1, 2))

            out = fprop_function(input_value)
            gmean, gvar = stats_function()
            ref = reference_function(normed_input)

            assert ng.testing.allclose(out,
                                       ref, atol=1e-5), '%e' % np.max(np.abs(out - ref))
            assert ng.testing.allclose(gmean,
                                       gmean_ref, atol=1e-6), '%e' % np.max(np.abs(gmean - gmean_ref))
            assert ng.testing.allclose(gvar,
                                       gvar_ref, atol=1e-6), '%e' % np.max(np.abs(gvar - gvar_ref))



