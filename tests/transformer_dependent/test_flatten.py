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

from ngraph.testing import check_derivative, RandomTensorGenerator
import numpy as np
import pytest
import ngraph as ng
rng = RandomTensorGenerator(0, np.float32)

pytestmark = pytest.mark.transformer_dependent


def test_flatten_deriv_simplified(transformer_factory):
    """
    Test derivative with dot and flatten
    """
    ax_N = ng.make_axis(length=3)
    ax_Y = ng.make_axis(length=2)

    x = ng.placeholder(ng.make_axes([ax_N]))
    w = ng.constant([5, 2], axes=ng.make_axes([ax_Y]))
    logits = ng.dot(x, w)
    cost = ng.sum(logits, reduction_axes=logits.axes)

    delta = 0.001
    u = rng.uniform(.1, 5.0, x.axes)
    check_derivative(cost, x, delta, u, atol=1e-2, rtol=1e-2)


@pytest.mark.xfail(strict=True)
def test_flatten_deriv(transformer_factory):
    from ngraph.frontends.neon import ax
    np.random.seed(0)

    # set shape
    C, D, H, W, N = (3, 1, 28, 28, 8)  # image
    Y = 10

    ax.C.length = C
    ax.D.length = D
    ax.H.length = H
    ax.W.length = W
    ax.N.length = N
    ax.Y.length = Y

    # conv output
    conv = ng.placeholder(ng.make_axes([ax.N, ax.H, ax.W, ax.C]))

    # flatten
    flatten = ng.flatten_at(conv, idx=1)
    num_flatten = flatten.axes.lengths[1]
    flatten = ng.cast_axes(flatten,
                           ng.make_axes([ax.N, ng.make_axis(num_flatten)]))

    # fc
    fc_weights_axes = ng.make_axes([ng.make_axis(num_flatten), ax.Y])
    fc_weights = ng.constant(np.random.randn(num_flatten, Y),
                             axes=fc_weights_axes)
    flatten_casted = ng.cast_axes(flatten,
                                  ng.make_axes([flatten.axes[0],
                                                fc_weights_axes[0] - 1]))
    logits = ng.dot(flatten_casted, fc_weights)
    cost = ng.sum(logits, reduction_axes=logits.axes)

    delta = 0.001
    u = rng.uniform(.1, 5.0, conv.axes)
    check_derivative(cost, conv, delta, u, atol=1e-2, rtol=1e-2)
