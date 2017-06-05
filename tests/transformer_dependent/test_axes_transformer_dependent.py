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
import numpy as np
import pytest

import ngraph as ng
from ngraph.testing import ExecutorFactory

pytestmark = pytest.mark.transformer_dependent


def test_idempotent_axes_a():
    """
    Test test axes transformations with autodiff, case a, reference test
    """
    with ExecutorFactory() as ex:
        axes = ng.make_axes([ng.make_axis(3), ng.make_axis(1)])

        w = ng.variable(axes, initial_value=np.ones((3, 1)))
        result = w + w

        result = ng.cast_axes(result, axes)
        cost = ng.sum(result, reduction_axes=axes)
        grad = ng.deriv(cost, w)

        grad_comp = ex.executor(grad)
        cost_comp = ex.executor(cost)

        cost_comp_val = cost_comp()
        grad_comp_val = grad_comp()
        grad_comp_np = np.ones((3, 1)) * 2.

        assert cost_comp_val == 6.0
        assert np.array_equal(grad_comp_val, grad_comp_np)


def test_idempotent_axes_b():
    """
    Test test axes transformations with autodiff, case b, with broadcast applied
    to the same tensor
    """
    with ExecutorFactory() as ex:
        axes = ng.make_axes([ng.make_axis(3), ng.make_axis(1)])

        w = ng.variable(axes, initial_value=np.ones((3, 1)))
        l = ng.broadcast(w, axes)
        r = ng.broadcast(w, axes)
        result = ng.add(l, r)

        result = ng.cast_axes(result, axes)
        cost = ng.sum(result, reduction_axes=axes)
        grad = ng.deriv(cost, w)

        grad_comp = ex.executor(grad)
        cost_comp = ex.executor(cost)

        assert cost_comp() == 6.0
        assert np.array_equal(grad_comp(), np.ones((3, 1)) * 2.)


def test_idempotent_axes_c():
    """
    Test test axes transformations with autodiff, case c, with broadcast,
    slice, cast and dim-shuffle
    """
    with ExecutorFactory() as ex:
        axes = ng.make_axes([ng.make_axis(3), ng.make_axis(1)])
        result_axes = [ng.make_axis(length=axis.length) for axis in axes]

        # variable
        w = ng.variable(axes, initial_value=np.ones((3, 1)))

        # broadcast l / r, introducing dummy length 1 axes
        l = ng.broadcast(w, axes)
        r = ng.broadcast(w, axes)

        # slice
        axes_slice = [slice(None, None, None), slice(None, None, None)]
        l_sliced = ng.tensor_slice(l, axes_slice)
        r_sliced = ng.tensor_slice(r, axes_slice)

        # cast r
        r_sliced_casted = ng.cast_axes(r_sliced, axes)

        # perform add
        result = ng.add(l_sliced, r_sliced_casted)

        # cast / dimshuffle
        result = ng.cast_axes(result, result_axes)
        result = ng.axes_with_order(result, result_axes)

        # cost and grad
        cost = ng.sum(result, reduction_axes=result.axes)
        grad = ng.deriv(cost, w)

        grad_comp = ex.executor(grad)
        cost_comp = ex.executor(cost)

        cost_comp_ng = cost_comp()
        grad_comp_ng = grad_comp()
        grad_comp_np = np.ones((3, 1)) * 2.
        assert cost_comp_ng == 6.0
        assert np.array_equal(grad_comp_ng, grad_comp_np)


def test_scalar_broadcast():
    """
    Test broadcasting a scalar into a tensor
    """
    with ExecutorFactory() as ex:
        x_axes = ng.make_axes()
        broadcast_axes = ng.make_axes([ng.make_axis(2), ng.make_axis(3)])
        x = ng.constant(1., axes=x_axes)
        z = ng.broadcast(x, axes=broadcast_axes)
        z_comp = ex.executor(z)
        assert np.array_equal(z_comp(), np.ones(broadcast_axes.lengths))
