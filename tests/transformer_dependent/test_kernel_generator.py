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
from __future__ import division
import numpy as np
import pytest
import ngraph as ng

from ngraph.testing import executor, ExecutorFactory, is_flex_factory
from ngraph.testing.random import RandomTensorGenerator

rng = RandomTensorGenerator()

pytestmark = pytest.mark.transformer_dependent


@pytest.fixture(scope='module', params=[
    (64, 10, 28, 32),
    (3, 16, 16, 4),
    (7, 15, 19, 5),
    (3, 5, 7, 2)])
def input_axes(request):
    return ng.make_axes([
        ng.make_axis(length=request.param[0]),
        ng.make_axis(length=request.param[1]),
        ng.make_axis(length=request.param[2]),
        ng.make_axis(length=request.param[3])
    ])


def test_exit_condition(transformer_factory):
    bsz = 16
    class_num = 10

    # Limiting maximum absolute value for tensors elements to 7.9.
    #
    # There is used np.random.randn function to fill tensors with random values. It can give any
    # value as a result however values above 5 are highly improbable and would appear very rarely.
    # Limit 7.9 would almost never modify the tested tensor but would prevent from random
    # failures from time to time when the test is run in continuous environment.
    # This limit is approximate upper bound of range [4, 8). Numbers from this region can be
    # expressed by flexpoint number of the same dec.
    # Why not 15.9 that is approximate limit of [8, 16) range ?
    # Numbers above 8 are highly improbable and if appear from time to time can cause random
    # failures due to reduced accuracy of all numbers in tensor. Most numbers in normal
    # distribution are close to 0.

    is_flex = is_flex_factory(transformer_factory)
    clip_val = 7.9 if is_flex else 0

    N, Y = ng.make_axis(bsz), ng.make_axis(class_num)
    y_val = rng.randn_abs_clip(ng.make_axes([N, Y]), clip_max=clip_val)
    y = ng.constant(y_val, ng.make_axes([N, Y]))

    likelihood = ng.log(ng.softmax(y, normalization_axes=y.axes[1]))

    with ExecutorFactory() as ex:
        comp = ex.executor(likelihood)

        val1 = comp()
        val2 = comp()
        ng.testing.assert_allclose(val1, val2, atol=0, rtol=0)


def test_4d_elementwise(transformer_factory, input_axes):

    # Limiting maximum absolute value for tensors elements to 7.9.
    # See description in function test_exit_condition above

    is_flex = is_flex_factory(transformer_factory)
    clip_val = 7.9 if is_flex else 0

    x_val = rng.randn_abs_clip(input_axes, clip_max=clip_val)
    y_val = rng.randn_abs_clip(input_axes, clip_max=clip_val)
    x = ng.constant(x_val, input_axes)
    y = ng.constant(y_val, input_axes)

    out = ng.add(x, y)

    with executor(out) as ex:
        graph_val = ex()

    np_val = np.add(x_val, y_val)

    ng.testing.assert_allclose(graph_val, np_val, rtol=1e-4)


def test_4d_reduction(transformer_factory, input_axes):

    # Limiting maximum absolute value for tensors elements to 7.9.
    # See description in function test_exit_condition above

    is_flex = is_flex_factory(transformer_factory)
    clip_val = 7.9 if is_flex else 0

    x_val = rng.randn_abs_clip(input_axes, clip_max=clip_val)
    x = ng.constant(x_val, input_axes)

    out1 = ng.sum(x, reduction_axes=input_axes[1])
    out2 = ng.sum(x, reduction_axes=input_axes[3])

    with executor([out1, out2]) as ex:
        graph_val1, graph_val2 = ex()
        np_val1 = np.sum(x_val, 1)
        np_val2 = np.sum(x_val, 3)
        ng.testing.assert_allclose(graph_val1, np_val1, rtol=1e-4, atol_multiplier=x_val.shape[1])
        ng.testing.assert_allclose(graph_val2, np_val2, rtol=1e-4, atol_multiplier=x_val.shape[3])


def test_4d_chained(transformer_factory, input_axes):

    # Limiting maximum absolute value for tensors elements to 7.9.
    # See description in function test_exit_condition above

    # Limitting minimum absolute value for tensors being input to reciprocal operation to 1/7.9
    #
    # This is consequence of the above and flexpoint accuracy.
    # Numbers very small have poor absolute accuracy. When reciprocal of them is calculated the
    # results becomes very large and has even worse accuracy. When small numbers would be accepted
    # as an input to reciprocal in the test the absolute maximum value of the result is undefined
    # and so absolute tolerance.
    # To have possibility to set atol in the test and test could pass with it minimum element of
    # the tensor that is input to reciprocal operation has to be limited.

    is_flex = is_flex_factory(transformer_factory)
    clip_val_max = 7.9 if is_flex else 0
    clip_val_min = 1.0 / 7.9 if is_flex else 0

    x_val = rng.randn_abs_clip(input_axes, clip_min=clip_val_min, clip_max=clip_val_max)
    y_val = rng.randn_abs_clip(input_axes, clip_max=clip_val_max)
    x = ng.constant(x_val, input_axes)
    y = ng.constant(y_val, input_axes)

    im = ng.reciprocal(x)
    out = ng.sum(ng.add(im, y), reduction_axes=input_axes[0])

    with executor(out) as ex:
        graph_val = ex()
    np_val = np.sum(np.add(np.reciprocal(x_val), y_val), 0)

    # atol_multiplier = 15 * x_val.shape[0]
    #
    # x_val.shape[0] is number elements added together in operation
    # ng.sum(X, reduction_axes=input_axes[0])
    #
    # 15 is calculated the following way:
    #
    # Input tensor has values from the range 1/7.9 - 7.9
    # For DEC=12 absolute error is equal to 0.5*2^-12 = 0.000122
    # 1/7.9 = 0.126582 with this error becomes 0.126704
    # Reciprocal of 1/7.9 is 7.9
    # Reciprocal of 1/7.9 + err = 7.892389
    # Absolute difference is 0.007611
    # It is 15.2 times larger then atol limit 5e-4 from Argon transformer
    ng.testing.assert_allclose(graph_val, np_val, rtol=1e-4, atol_multiplier=15 * x_val.shape[0])


def test_kernel_cache(transformer_factory):
    X = ng.make_axis(32)
    Y = ng.make_axis(32)
    C = ng.make_axis(16384)
    axes = ng.make_axes([
        X,
        Y
    ])
    bcast_axes = ng.make_axes([
        X,
        Y,
        C
    ])

    # Limiting maximum absolute value for tensors elements to 7.9.
    # See description in function test_exit_condition above

    is_flex = is_flex_factory(transformer_factory)
    clip_val = 7.9 if is_flex else 0

    x_val = rng.randn_abs_clip(axes, clip_max=clip_val)
    y_val = rng.randn_abs_clip(bcast_axes, clip_max=clip_val)
    z_val = rng.randn_abs_clip(bcast_axes, clip_max=clip_val)

    x = ng.constant(x_val, axes)
    y = ng.constant(y_val, bcast_axes)
    z = ng.constant(z_val, bcast_axes)

    out = ng.add(ng.add(x, y), z)

    with executor(out) as ex:
        graph_val = ex()
    np_val = np.add(np.add(x_val.reshape(32, 32, 1), y_val), z_val)
    ng.testing.assert_allclose(graph_val, np_val, rtol=1e-4, atol_multiplier=2)
