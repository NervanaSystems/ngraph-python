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
from ngraph.frontends.neon.axis import make_shadow_axis
from ngraph.testing import ExecutorFactory

pytestmark = pytest.mark.transformer_dependent


@pytest.fixture(scope='module', params=[2])
def input_size(request):
    return request.param


@pytest.fixture(scope='module')
def feature_axis(input_size):
    return ng.make_axis(input_size, name='Fin')


@pytest.fixture(scope='module', params=[3])
def batch_axis(request):
    return ng.make_axis(request.param, name='N')


@pytest.fixture(scope='module', params=[1, 5])
def output_size(request):
    return request.param


@pytest.fixture(scope='module')
def input_placeholder(feature_axis, batch_axis):
    return ng.placeholder([feature_axis, batch_axis])


@pytest.config.argon_disabled  # TODO triage
def test_linear_zeros(input_placeholder, output_size, transformer_factory):
    # basic sanity check with 0 weights random inputs
    x = np.random.random(input_placeholder.axes.lengths)
    layer = Linear(nout=output_size, init=UniformInit(0.0, 0.0))

    with ExecutorFactory() as ex:
        if ex.transformer.transformer_name == 'hetr':
            pytest.xfail("hetr fork-safe issue on mac")
        comp = ex.executor(layer(input_placeholder), input_placeholder)
        output_values = comp(x)

    assert np.min(output_values) == 0.0 and np.max(output_values) == 0.0


@pytest.config.argon_disabled  # TODO triage
def test_linear_ones(input_size, input_placeholder, output_size, transformer_factory):
    # basic sanity check with all ones on the inputs and weights, check that
    # each row in output is the sum of the weights for that output this check
    # will confirm that the correct number of operations is being run
    x = np.ones(input_placeholder.axes.lengths)
    layer = Linear(nout=output_size, init=UniformInit(1.0, 1.0))

    with ExecutorFactory() as ex:
        if ex.transformer.transformer_name == 'hetr':
            pytest.xfail("hetr fork-safe issue on mac")
        out = layer(input_placeholder)
        comp = ex.executor([out, layer.W], input_placeholder)
        output_values, w = comp(x)

    assert ng.testing.allclose(
        np.ones(out.axes.lengths) * input_size,
        output_values,
        atol=0.0, rtol=0.0
    )


def test_linear_axes_nout():
    feature_axis = ng.make_axis(1, name='A')
    batch_axis = ng.make_axis(2, name='N')

    x = ng.placeholder([feature_axis, batch_axis])
    linear = Linear(nout=3, init=UniformInit(1.0, 1.0))(x)

    assert feature_axis not in linear.axes
    assert batch_axis in linear.axes
    assert linear.axes.batch_axis().length == 2
    assert linear.axes.sample_axes().lengths == (3,)


def test_linear_W_axes_nout():
    feature_axis = ng.make_axis(1, name='A')
    batch_axis = ng.make_axis(2, name='N')

    x = ng.placeholder([feature_axis, batch_axis])
    linear = Linear(nout=3, init=UniformInit(1.0, 1.0))
    linear(x)

    assert linear.W.axes.batch_axis() is None
    assert feature_axis in linear.W.axes
    assert len(linear.W.axes - feature_axis) == 1
    assert (linear.W.axes - feature_axis)[0].length == 3


def test_linear_accepts_axes_axis():
    """ Ensure that Linear.__init__ accepts an Axis as axes """
    Linear(axes=ng.make_axis(1), init=UniformInit(1.0, 1.0))


def test_linear_axes():
    feature_axis = ng.make_axis(1, name='A')
    batch_axis = ng.make_axis(2, name='N')
    feature_out_axis = ng.make_axis(3, name='Aout')

    x = ng.placeholder([feature_axis, batch_axis])
    linear = Linear(axes=feature_out_axis, init=UniformInit(1.0, 1.0))(x)

    assert feature_axis not in linear.axes
    assert batch_axis in linear.axes
    assert linear.axes.batch_axis().length == 2
    assert linear.axes.sample_axes().lengths == (3,)


def test_linear_invalid_shadow_axes():
    with pytest.raises(ValueError):
        Linear(axes=make_shadow_axis(ng.make_axis(1, name='A')), init=UniformInit(1.0, 1.0))


def test_linear_invalid_recurrent_axes():
    with pytest.raises(ValueError):
        Linear(axes=ng.make_axis(1, name='REC'), init=UniformInit(1.0, 1.0))


def test_linear_invalid_batch_axes():
    with pytest.raises(ValueError):
        Linear(axes=ng.make_axis(1, name='N'), init=UniformInit(1.0, 1.0))
