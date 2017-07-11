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


@pytest.fixture(params=[(1, 2, 1),
                        (2, 3, 2),
                        (15, 5, 1)])
def concatenate_variables(request):
    num_vars, num_axes, concat_pos = request.param
    common_axes = [ng.make_axis(length=2) for _ in range(num_axes - 1)]
    x_list = list()
    np_list = list()
    ax = ng.make_axis(length=np.random.randint(3, 10))
    axes = ng.make_axes(common_axes[:concat_pos] + [ax] + common_axes[concat_pos:])
    for _ in range(num_vars):
        var = np.random.uniform(0, 1, axes.full_lengths)
        np_list.append(var)
        x_list.append(ng.constant(var, axes=axes))

    return x_list, np_list, concat_pos


@pytest.fixture()
def C():
    return ng.make_axis(length=200)


@pytest.fixture()
def N():
    return ng.make_axis(length=1)


@pytest.fixture()
def M():
    return ng.make_axis(length=3)


def test_sequential(N):
    x = ng.variable([N], initial_value=0)
    x0 = x + x
    x1 = x + x
    p = ng.sequential([
        x0,
        ng.assign(x, 2),
        x1,
        x0
    ])
    with ExecutorFactory() as ex:
        x0_val, x1_val, p_val = ex.executor([x0, x1, p])()
    assert x0_val == 0
    assert x1_val == 4
    assert p_val == 0


def test_sequential_reduce(M):
    x = ng.variable([M], initial_value=1)
    x0 = x + x
    x1 = ng.sum(x0, out_axes=())
    x2 = ng.sum(x0, out_axes=()) + x0
    p = ng.sequential([
        x0,
        x1,
        x2
    ])

    with ExecutorFactory() as ex:
        x0_val, x1_val, x2_val, p_val, x_val = ex.executor([x0, x1, x2, p, x])()
        x0_np = x_val + x_val
        x1_np = np.sum(x0_np)
        x2_np = x1_np + x0_np
        assert np.allclose(x0_val, x0_np)
        assert np.allclose(x1_val, x1_np)
        assert np.allclose(x2_val, x2_np)
        assert np.allclose(p_val, x2_np)


def test_sequential_side(M):
    x1_np = 2
    x2_np = 3
    b_np = 1
    x_np = np.array([1, 2, 3], dtype=np.float32)

    x = ng.variable([M], initial_value=x_np)
    x1 = ng.persistent_tensor(axes=(), initial_value=x1_np)
    x2 = ng.persistent_tensor(axes=(), initial_value=x2_np)
    x1_vo = ng.value_of(x1)
    x2_vo = ng.value_of(x2)
    b = ng.persistent_tensor(axes=(), initial_value=b_np)

    y = ng.sequential([
        x1_vo,
        x2_vo,
        ng.assign(x1, ng.sum(x, out_axes=()) + x1 * b + (1 - b)),
        ng.assign(x2, ng.mean(x, out_axes=()) + x2 * b + (1 - b)),
        x * 2
    ])

    with ExecutorFactory() as ex:
        main_effect = ex.executor((y, x1_vo, x2_vo, x1, x2))
        current_values = ex.executor((x1, x2))

        # Run main path #1
        y_val, x1_init_val, x2_init_val, x1_final_val, x2_final_val = main_effect()
        y_np = x_np * 2

        assert np.allclose(y_val, y_np)
        assert np.allclose(x1_init_val, x1_np)
        assert np.allclose(x2_init_val, x2_np)
        x1_np = np.sum(x_np) + x1_np * b_np + (1 - b_np)
        x2_np = np.mean(x_np) + x2_np * b_np + (1 - b_np)
        assert np.allclose(x1_final_val, x1_np)
        assert np.allclose(x2_final_val, x2_np)

        x1_val, x2_val = current_values()
        assert np.allclose(x1_val, x1_np)
        assert np.allclose(x2_val, x2_np)

        # Run main path #2 (Should be the same as before)
        y_val, x1_init_val, x2_init_val, x1_final_val, x2_final_val = main_effect()
        y_np = x_np * 2

        assert np.allclose(y_val, y_np)
        assert np.allclose(x1_init_val, x1_np)
        assert np.allclose(x2_init_val, x2_np)
        x1_np = np.sum(x_np) + x1_np * b_np + (1 - b_np)
        x2_np = np.mean(x_np) + x2_np * b_np + (1 - b_np)
        assert np.allclose(x1_final_val, x1_np)
        assert np.allclose(x2_final_val, x2_np)


@pytest.mark.flex_disabled
@pytest.config.argon_disabled  # TODO triage
def test_concatenate(transformer_factory, concatenate_variables):
    x_list, np_list, pos = concatenate_variables

    with ExecutorFactory() as ex:
        v = ng.concat_along_axis(x_list, x_list[0].axes[pos])
        d = ng.deriv(v, x_list[0],
                     error=ng.constant(np.ones(v.axes.lengths), axes=v.axes))
        f = ex.executor([v, d])
        e_v, e_d = f()
        np_v = np.concatenate(np_list, axis=pos)
        assert ng.testing.allclose(e_v.copy(), np_v)
        assert ng.testing.allclose(e_d.copy(), np.ones(x_list[0].axes.lengths))


@pytest.mark.flex_disabled
@pytest.config.argon_disabled  # TODO triage
def test_concat_different_axis_lengths(transformer_factory):
    ax1 = ng.make_axis(length=3, name="concat")
    ax2 = ng.make_axis(length=2, name="concat")
    ax3 = ng.make_axis(length=10, name="other")

    x = ng.placeholder(axes=[ax1, ax3])
    y = ng.placeholder(axes=[ax2, ax3])

    np_x = np.zeros(x.axes.lengths)
    np_y = np.zeros(y.axes.lengths)

    # ax1 and ax2 have same name, so this should work
    v = ng.concat_along_axis([x, y], ax1)
    with ExecutorFactory() as ex:
        f = ex.executor(v, x, y)
        e_v = f(np_x, np_y)
        np_v = np.concatenate([np_x, np_y], axis=0)
        assert ng.testing.allclose(e_v.copy(), np_v)


@pytest.mark.flex_disabled
def test_variable_init(transformer_factory, C):
    w_init = np.random.rand(C.length)
    W = ng.variable(ng.make_axes([C]), initial_value=w_init)

    with ExecutorFactory() as ex:
        result = ex.executor(W)()
    ng.testing.assert_allclose(result, w_init)


@pytest.mark.flex_disabled
def test_initial_value(transformer_factory):
    # Test work-around for issue #1138
    w = [3, 4, 5]
    x = ng.constant(w)
    y = ng.variable([ng.make_axis(length=len(w))], initial_value=x)
    with ExecutorFactory() as ex:
        result = ex.executor(y)()
    ng.testing.assert_allclose(result, np.asarray(w, dtype=np.float32))


@pytest.config.argon_disabled  # TODO triage
def test_multiple_computations(transformer_factory):
    """
    Create multiple computations for the same value.

    Args:
        transformer_factory:

    Returns:

    """
    C = ng.make_axis(length=2)
    D = ng.make_axis(length=3)

    x = ng.placeholder([C, D])

    x0_slice = x[0, :]
    x1_slice = x[1, :]

    y1 = x0_slice * 2 + x1_slice * 3

    x_np = np.array([[10, 20, 30], [1, 2, 3]], dtype='float32')
    y1_np = x_np[0, :] * 2 + x_np[1, :] * 3

    with ExecutorFactory() as ex:
        fs = [ex.executor(y1, x) for i in range(5)]
        vals_np = [y1_np for f in fs]
        vals = [f(x_np) for f in fs]
        # print(vals_np)
        # print(vals)
        assert ng.testing.allclose(vals, vals_np)
