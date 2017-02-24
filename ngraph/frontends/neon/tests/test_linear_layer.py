# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
Test of the mlp/linear layer
'''
import pytest
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Linear, UniformInit
from ngraph.testing import ExecutorFactory


@pytest.fixture(scope='module', params=[1, 2, 1023, 1024, 1025])
def feature_axis(request):
    return ng.make_axis(request.param)


@pytest.fixture(scope='module', params=[128])
def batch_axis(request):
    return ng.make_axis(request.param, name='N')


@pytest.fixture(scope='module', params=[1, 4, 1023, 1024, 1025])
def output_size(request):
    return request.param


@pytest.fixture(scope='module')
def input_placeholder(feature_axis, batch_axis):
    return ng.placeholder([feature_axis, batch_axis])


def test_linear_zeros(input_placeholder, output_size, transformer_factory):
    # basic sanity check with 0 weights random inputs
    x = np.random.random(input_placeholder.axes.lengths)
    layer = Linear(nout=output_size, init=UniformInit(0.0, 0.0))

    with ExecutorFactory() as ex:
        comp = ex.executor(layer.train_outputs(input_placeholder),
                           input_placeholder)
        output_values = comp(x)

    assert np.min(output_values) == 0.0 and np.max(output_values) == 0.0


def test_linear_ones(input_placeholder, output_size, transformer_factory):
    # basic sanity check with all ones on the inputs and weights, check that
    # each row in output is the sum of the weights for that output this check
    # will confirm that the correct number of operations is being run
    x = np.ones(input_placeholder.axes.lengths)
    layer = Linear(nout=output_size, init=UniformInit(1.0, 1.0))

    with ExecutorFactory() as ex:
        comp = ex.executor([layer.train_outputs(input_placeholder), layer.W],
                           input_placeholder)
        output_values, w = comp(x)

    sums = np.sum(w, axis=1, keepdims=True) * np.ones((1, x.shape[1]))

    assert ng.testing.allclose(sums, output_values, atol=0.0, rtol=0.0)
