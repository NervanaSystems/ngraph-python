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
from builtins import range

import numpy as np
import pytest
import ngraph as ng
import collections
from ngraph.testing import check_derivative, ExecutorFactory, \
    RandomTensorGenerator, numeric_derivative, executor

pytestmark = pytest.mark.transformer_dependent
pytest.mark.argon_disab_loose = pytest.mark.xfail(pytest.config.getvalue("transformer") == "argon",
                                                  reason="Not supported by argon backend",
                                                  strict=False)


rng = RandomTensorGenerator(0, np.float32)


def test_constant_multiply(transformer_factory):
    # TODO: better error message when missing axes length in cases where it
    # is needed
    Y = ng.make_axis(length=1)

    # TODO: don't require axes
    a = ng.constant(np.array([4.0], dtype='float32'), [Y])
    b = ng.constant(np.array([2.0], dtype='float32'), [Y])

    c = ng.multiply(a, b)

    with executor(c) as ex:
        result = ex()
    ng.testing.assert_allclose(result, [8])


def test_constant_tensor_multiply(transformer_factory):
    Y = ng.make_axis(length=2)
    N = ng.make_axis(length=2)

    a = ng.constant(np.array([[1.0, 1.0], [1.0, 1.0]], dtype='float32'), [Y, N])
    b = ng.constant(np.array([[1.0, 1.0], [1.0, 1.0]], dtype='float32'), [Y, N])

    c = ng.multiply(a, b)

    with executor(c) as ex:
        result = ex()
        ng.testing.assert_allclose(result, [[1.0, 1.0], [1.0, 1.0]])


def test_tensor_sum_single_reduction_axes(transformer_factory):
    """TODO."""
    Y = ng.make_axis(length=2)
    N = ng.make_axis(length=2)

    a = ng.constant(np.array([[1.0, 1.0], [1.0, 1.0]], dtype='float32'), [N, Y])

    b = ng.sum(a, reduction_axes=Y)

    with executor(b) as ex:
        result = ex()
        ng.testing.assert_allclose(result, [2.0, 2.0])


def test_scalar(transformer_factory):
    """TODO."""
    # Simple evaluation of a scalar
    val = 5
    x = ng.constant(val)

    with executor(x) as ex:
        cval = ex()
    assert cval.shape == ()
    ng.testing.assert_allclose(cval, val)


def test_tensor_constant(transformer_factory):
    W = ng.make_axis().named('W')
    H = ng.make_axis().named('H')

    # Pass a NumPy array through as a constant
    W.length = 10
    H.length = 20
    aaxes = ng.make_axes([W, H])
    ashape = aaxes.lengths
    asize = aaxes.size
    aval = np.arange(asize, dtype=np.float32).reshape(ashape)

    x = ng.constant(aval, aaxes)
    with executor(x) as ex:
        cval = ex()
    ng.testing.assert_allclose(cval, aval)


@pytest.config.flex_disabled(reason='Results mismatch')
def test_placeholder(transformer_factory):
    W = ng.make_axis(length=10)
    H = ng.make_axis(length=20)

    # Pass array through a placeholder
    aaxes = ng.make_axes([W, H])
    ashape = aaxes.lengths
    asize = aaxes.size
    aval = np.arange(asize, dtype=np.float32).reshape(ashape)

    x = ng.placeholder(aaxes)
    d = 2 * x
    d2 = ng.squared_L2(x, out_axes=None)

    with ExecutorFactory() as ex:
        # Return placeholder, param is placeholder
        placeholder_fun = ex.executor(x, x)
        prod_fun = ex.executor([d, d2], x)

        cval = placeholder_fun(aval)
        ng.testing.assert_allclose(cval, aval)

        # Pass a different array though
        u = rng.uniform(-1.0, 1.0, aaxes)
        cval = placeholder_fun(u)
        ng.testing.assert_allclose(cval, u)

        cval, s = prod_fun(aval)
        ng.testing.assert_allclose(cval, aval * 2)
        ng.testing.assert_allclose(s[()], np.dot(aval.flatten(), aval.flatten()))

        cval, s = prod_fun(u)
        u2 = u * 2
        ng.testing.assert_allclose(cval, u2)
        ng.testing.assert_allclose(s[()], np.dot(u.flatten(), u.flatten()))


@pytest.fixture(params=['sum', 'prod', 'max', 'min'])
def reduction(request):
    return request.param


@pytest.fixture(params=[
    slice(0, 1, None),
    slice(1, 2, None),
    slice(2, None, None),
    slice(0, 2, None),
    slice(1, None, None)
])
def sub_axes(request):
    return request.param


# TODO this is a non-strict disable since not all parametrizations fail argon
@pytest.mark.argon_disab_loose  # TODO Triage
def test_reduction(transformer_factory, reduction, sub_axes):
    axes = ng.make_axes([ng.make_axis(length=4),
                         ng.make_axis(length=4),
                         ng.make_axis(length=4)])

    u = rng.uniform(-1.0, 1.0, axes)

    npred = getattr(np, reduction)
    bered = getattr(ng, reduction)
    reduction_axes = axes[sub_axes]

    p_u = ng.placeholder(axes)
    dims = tuple(axes.index(axis) for axis in reduction_axes)
    npval = npred(u, dims)
    graph_reduce = bered(p_u, reduction_axes=reduction_axes)
    with executor(graph_reduce, p_u) as ex:
        graph_val = ex(u)
        ng.testing.assert_allclose(
            npval, graph_val, rtol=1e-5), 'red:{red}, axes:{axes}'.format(
            red=reduction, axes=reduction_axes)


@pytest.config.argon_disabled  # TODO Triage
def test_reduction_deriv(transformer_factory, reduction, sub_axes):
    if reduction in ('max', 'min'):
        pytest.skip("max/min needed to be tested differently")
    if sub_axes in (slice(0, 2, None), slice(1, None, None)) and reduction == "prod":
        pytest.config.flex_skip_now("Too big values for Flex ( > 32767 )")
    axes = ng.make_axes([ng.make_axis(length=4),
                         ng.make_axis(length=10),
                         ng.make_axis(length=10)])

    delta = .001

    u = rng.discrete_uniform(1.0, 2.0, 2 * delta, axes)

    bered = getattr(ng, reduction)
    reduction_axes = axes[sub_axes]

    # Need to test max/min differently since if two elements are extremums
    # and we modify one, the derivative will change.
    p_u = ng.placeholder(axes)
    graph_reduce = bered(p_u, reduction_axes=reduction_axes)

    check_derivative(graph_reduce, p_u, delta, u, atol=1e-1, rtol=1e-1)


@pytest.fixture(params=[
    (0, ["A0"]),
    (1, ["A1"]),
    (2, ["A2"]),
    ((0, 1), ["A0", "A1"]),
    ((0, 2), ["A0", "A2"]),
    ((1, 2), ["A1", "A2"]),
    pytest.config.flex_disabled(
        ((0, 1, 2), ["A0", "A1", "A2"]),
        reason="Too big values for Flex ( > 32767 )"
    )
])
def prod_constant(request):
    axes_dict = collections.OrderedDict([
        ("A0", ng.make_axis(length=2)),
        ("A1", ng.make_axis(length=3)),
        ("A2", ng.make_axis(length=4))
    ])
    np_axis, axes_names = request.param
    return np_axis, map(lambda x: axes_dict[x], axes_names), axes_dict.values()


@pytest.config.argon_disabled  # TODO Triage
def test_prod_constant(transformer_factory, prod_constant):
    """
    Test reduce product of constants
    """
    np_axis, ng_axis, axes_values = prod_constant

    # ngrpah op
    const_3d = ng.broadcast(ng.constant(2., axes=[]), axes=axes_values)
    prod = ng.prod(const_3d, reduction_axes=ng_axis)

    # numpy results
    np_const_3d = np.ones((2, 3, 4)) * 2.

    res_np = np.prod(np_const_3d, axis=np_axis)

    # define comp
    with ExecutorFactory() as ex:
        comps = ex.executor(prod)
        res_ng = comps()

    np.testing.assert_allclose(res_np, res_ng)


@pytest.fixture(params=[
    pytest.config.flex_disabled(
        np.array([[[1., 2., 3.], [4., 5., 0.], [0., 6., 0.]],
                  [[1., 2., 3.], [4., 5., 6.], [7., 8., 0.]]]),
        reason="Too big values for Flex ( > 32767 )"),
    np.array([[1., 2., 3.], [4., 5., 0.], [0., 6., 0.]]),
    np.array([1., 2., 3.]),
    np.array([0., 2., 3.]),
    np.array([0., 0., 3.]),
    np.array([0., 0., 0.]),
    np.array([0.]),
    np.array([2.]),
    np.array(0.),
    np.array(2.),
])
def prod_deriv_arrays(request):
    return request.param


@pytest.config.argon_disabled  # TODO Triage
def test_prod_deriv(transformer_factory, prod_deriv_arrays):
    """
    Test reduce product's gradient
    """
    def power_set(lst):
        """
        power_set([0, 1, 2]) is:
        [[], [0], [1], [0, 1], [2], [0, 2], [1, 2], [0, 1, 2]]
        """
        result = [[]]
        for x in lst:
            result.extend([subset + [x] for subset in result])
        return result

    def get_all_reduction_axes(axes):
        """
        Get all possible reduction axes
        """
        ndim = len(axes.lengths)
        if ndim == 0:
            return axes
        else:
            results = []
            all_indices = power_set(range(ndim))
            for indices in all_indices:
                if not indices:
                    results.append(ng.make_axes([]))
                else:
                    results.append(ng.make_axes([axes[index] for index in indices]))
            return results

    def shape_to_axes(shape):
        """
        Convert shape to axes
        """
        if not shape:
            return ng.make_axes()
        axes = ng.make_axes([ng.make_axis(length=s) for s in shape])
        return axes

    x_val = prod_deriv_arrays
    axes = shape_to_axes(x_val.shape)
    all_reduction_axes = get_all_reduction_axes(axes)
    for reduction_axes in all_reduction_axes:
        x = ng.placeholder(axes=axes)
        x_prod = ng.prod(x, reduction_axes)
        check_derivative(x_prod, x, 0.001, x_val, atol=1e-3, rtol=1e-3)


def test_reciprocal(transformer_factory, input_tensor):
    """TODO."""
    p_u = input_tensor
    u = rng.uniform(.1, 5.0, p_u.axes)

    rec_u_np = np.reciprocal(u)
    rec_u = ng.reciprocal(p_u)

    with ExecutorFactory() as ex:
        rec_u_graph = ex.executor(rec_u, p_u)(u)
    ng.testing.assert_allclose(rec_u_np, rec_u_graph)


def test_reciprocal_derivative(transformer_factory, input_tensor):
    """TODO."""
    p_u = input_tensor

    delta = .001

    u = rng.uniform(.1, 5.0, p_u.axes)

    rec_u = ng.reciprocal(p_u)

    check_derivative(rec_u, p_u, delta, u, atol=1e-2, rtol=1e-2)


@pytest.fixture
def symmetric_tensor():
    axes = ng.make_axes([ng.make_axis(length=10), ng.make_axis(length=10)])
    return ng.placeholder(axes)


@pytest.fixture(scope="module")
def batch_axis(request):
    return ng.make_axis(length=request.config.getoption("--batch_size"),
                        name='N')


@pytest.fixture(scope="module")
def feature_axis():
    return ng.make_axis(length=3)


@pytest.fixture(scope="module")
def recurrent_axis():
    return ng.make_axis(length=4, name='REC')


@pytest.fixture(scope="module")
def input_axes(feature_axis, batch_axis):
    return ng.make_axes([feature_axis, batch_axis])


@pytest.fixture
def input_tensor(input_axes):
    return ng.placeholder(input_axes)


@pytest.fixture
def recurrent_input_tensor(feature_axis, recurrent_axis, batch_axis):
    axes = ng.make_axes([feature_axis, batch_axis, recurrent_axis])
    return ng.placeholder(axes)


@pytest.fixture(params=['add', 'subtract', 'multiply', 'divide'])
def elementwise_binary_op(request):
    return request.param


@pytest.fixture(params=['exp', 'log', 'tanh'])
def elementwise_unary_op(request):
    return request.param


def test_elementwise_binary_ops_matched_args(
    transformer_factory,
    elementwise_binary_op,
    symmetric_tensor
):
    """TODO."""
    np_op = getattr(np, elementwise_binary_op)
    be_op = getattr(ng, elementwise_binary_op)
    p_u = symmetric_tensor
    p_v = ng.placeholder(p_u.axes)

    u = rng.uniform(-1.0, 1.0, p_u.axes)
    v = rng.uniform(1.0, 2.0, p_v.axes)

    compare_f_at_x(
        be_op(p_u, p_v), [p_u, p_v],
        np_op, [u, v],
        atol=1e-4, rtol=1e-4
    )


def test_elementwise_binary_ops_matched_args_deriv_lhs(
    transformer_factory,
    elementwise_binary_op,
    symmetric_tensor
):
    """TODO."""
    be_op = getattr(ng, elementwise_binary_op)
    p_u = symmetric_tensor
    p_v = ng.placeholder(p_u.axes)

    u = rng.uniform(-1.0, 1.0, p_u.axes)
    v = rng.uniform(1.0, 2.0, p_v.axes)

    check_derivative(
        be_op(p_u, p_v), p_u, 0.001, u,
        parameters=[p_v],
        parameter_values=[v],
        atol=1e-4, rtol=1e-4,
    )


def test_elementwise_binary_ops_matched_args_deriv_rhs(
    transformer_factory,
    elementwise_binary_op,
    symmetric_tensor
):
    """TODO."""
    be_op = getattr(ng, elementwise_binary_op)
    p_u = symmetric_tensor
    p_v = ng.placeholder(p_u.axes)

    u = rng.uniform(-1.0, 1.0, p_u.axes)
    v = rng.uniform(1.0, 2.0, p_v.axes)

    check_derivative(
        be_op(p_u, p_v), p_v, 0.001, v,
        parameters=[p_u],
        parameter_values=[u],
        atol=1e-3, rtol=1e-3,
    )


def test_elementwise_unary_ops_matched_args(
    transformer_factory,
    elementwise_unary_op,
    symmetric_tensor
):
    """TODO."""
    delta = .001
    np_op = getattr(np, elementwise_unary_op)
    be_op = getattr(ng, elementwise_unary_op)

    p_u = symmetric_tensor
    u = rng.uniform(1.0, 2.0, p_u.axes)
    u_np = np_op(u)
    result_op = be_op(p_u)

    with ExecutorFactory() as ex:
        fun = ex.executor(result_op, p_u)
        dudunum_fun = ex.numeric_derivative(result_op, p_u, delta)
        dudut_fun = ex.derivative(result_op, p_u)

        u_t = fun(u)
        ng.testing.assert_allclose(u_np, u_t, atol=1e-4, rtol=1e-4)
        dudunum = dudunum_fun(u)
        dudut = dudut_fun(u)
        ng.testing.assert_allclose(dudunum, dudut, atol=1e-3, rtol=1e-3)


def test_elementwise_ops_unmatched_args(
    transformer_factory,
    elementwise_binary_op,
    batch_axis
):
    """TODO."""
    W = ng.make_axis(length=5)
    H = ng.make_axis(length=5)
    N = batch_axis

    broadcast_dims = (W.length, H.length, 1)

    np_op = getattr(np, elementwise_binary_op)
    be_op = getattr(ng, elementwise_binary_op)

    # Matched sizes
    p_u = ng.placeholder([W, H])
    p_v = ng.placeholder([W, H, N])
    u = rng.uniform(1.0, 2.0, p_u.axes)
    v = rng.uniform(1.0, 2.0, p_v.axes)

    # u op v
    uv_np = np_op(u.reshape(broadcast_dims), v)
    uv_op = be_op(p_u, p_v)

    with ExecutorFactory() as ex:

        # fun(u, v)
        uv_fun = ex.executor(uv_op, p_u, p_v)
        duvdunum_fun = ex.numeric_derivative(uv_op, p_u, .001, p_v)
        duvdut_fun = ex.derivative(uv_op, p_u, p_v)
        duvdvnum_fun = ex.numeric_derivative(uv_op, p_v, .001, p_u)
        duvdvt_fun = ex.derivative(uv_op, p_v, p_u)

        # fun(v, u)
        vu_np = np_op(v, u.reshape(broadcast_dims))
        vu_op = be_op(p_v, p_u)

        vu_fun = ex.executor(vu_op, p_u, p_v)
        dvudunum_fun = ex.numeric_derivative(vu_op, p_u, .001, p_v)
        dvudut_fun = ex.derivative(vu_op, p_u, p_v)
        dvudvnum_fun = ex.numeric_derivative(vu_op, p_v, .001, p_u)
        dvudvt_fun = ex.derivative(vu_op, p_v, p_u)

        # u op v
        result_be = uv_fun(u, v)
        ng.testing.assert_allclose(uv_np, result_be, atol=1e-4, rtol=1e-4)
        duvdunum = duvdunum_fun(u, v)
        duvdut = duvdut_fun(u, v)
        ng.testing.assert_allclose(duvdunum, duvdut, atol=1e-3, rtol=1e-3)

        duvdvnum = duvdvnum_fun(v, u)
        duvdvt = duvdvt_fun(v, u)
        ng.testing.assert_allclose(duvdvnum, duvdvt, atol=1e-3, rtol=1e-3)

        # v op u

        result_be = vu_fun(u, v)
        ng.testing.assert_allclose(vu_np, result_be, atol=1e-4, rtol=1e-4)
        dvudunum = dvudunum_fun(u, v)
        dvudut = dvudut_fun(u, v)
        ng.testing.assert_allclose(dvudunum, dvudut, atol=1e-3, rtol=1e-3)

        dvudvnum = dvudvnum_fun(v, u)
        dvudvt = dvudvt_fun(v, u)
        ng.testing.assert_allclose(dvudvnum, dvudvt, atol=1e-3, rtol=1e-3)


def np_softmax(x, axis):
    """
    TODO.

    Arguments:
      x: TODO
      axis: TODO

    Returns:
      TODO
    """
    # Shape for broadcasts
    shape = list(x.shape)
    shape[axis] = 1

    exps = np.exp(x - np.max(x, axis).reshape(shape))
    return exps / np.sum(exps, axis).reshape(shape)


def cross_entropy_binary_logistic(x, t):
    """
    TODO.

    Arguments:
      x: TODO
      t: TODO

    Returns:
      TODO
    """
    y = 1.0 / (1.0 + np.exp(-x))
    return -(np.log(y) * t + np.log(1 - y) * (1 - t))


def cross_entropy_binary_logistic_shortcut(x, t):
    """
    TODO.

    Arguments:
      x: TODO
      t: TODO

    Returns:
      TODO
    """
    y = 1.0 / (1.0 + np.exp(-x))
    return (1.0 - t) * x - np.log(y)


def test_cross_entropy_binary_logistic_shortcut(
    transformer_factory,
    input_tensor,
):
    """TODO."""
    p_u = input_tensor
    p_v = ng.placeholder(p_u.axes)
    u = rng.uniform(-3.0, 3.0, p_u.axes)
    v = np_softmax(rng.uniform(-3.0, 3.0, p_u.axes), 0)

    cel = cross_entropy_binary_logistic(u, v)
    cel_shortcut = cross_entropy_binary_logistic_shortcut(u, v)
    ng.testing.assert_allclose(cel, cel_shortcut, rtol=1e-5)

    with executor(ng.cross_entropy_binary_inner(ng.sigmoid(p_u), p_v), p_u, p_v) as ex:
        cel_graph = ex(u, v)
    ng.testing.assert_allclose(cel, cel_graph, rtol=1e-5)


def test_cross_entropy_binary(
    transformer_factory,
    input_tensor
):
    """TODO."""
    p_u = input_tensor
    p_v = ng.placeholder(p_u.axes)

    u = rng.uniform(-3.0, 3.0, p_u.axes)
    v = rng.uniform(-3.0, 3.0, p_u.axes)

    delta = .001

    y = ng.sigmoid(p_u)
    t = ng.softmax(p_v)
    val_u = ng.cross_entropy_binary_inner(y, t)

    with ExecutorFactory() as ex:
        dval_u_num_fun = ex.numeric_derivative(val_u, p_u, delta, p_v)
        dval_u_graph_fun = ex.derivative(val_u, p_u, p_v)

        dval_u_num = dval_u_num_fun(u, v)
        dval_u_graph = dval_u_graph_fun(u, v)
        ng.testing.assert_allclose(dval_u_graph, dval_u_num, atol=1e-2, rtol=1e-2)


def test_cross_entropy_binary_unmatched_axes(input_tensor):
    """If y and t have different axes, an error should be thrown immediately"""
    y = input_tensor
    feature_axis, batch_axis = y.axes
    t = ng.placeholder([ng.make_axis(feature_axis.length), batch_axis])

    with pytest.raises(ng.UnmatchedAxesError):
        ng.cross_entropy_binary_inner(y, t)


def adiff_softmax(x):
    """
    The version of the diff we use in autodiff, without batch axis.

    Arguments:
      x: return:

    Returns:
      TODO
    """

    def softmax_adiff(y_, y):
        """
        TODO.

        Arguments:
          y_: TODO
          y: TODO

        Returns:
          TODO
        """
        z = y_ * y
        zs = z.sum()
        x_ = z - zs * y
        return x_

    y = np_softmax(x, 0)
    n = x.shape[0]
    result = np.zeros((n, n))
    y_ = np.zeros_like(x)
    for i in range(n):
        y_[i] = 1
        result[i, :] = softmax_adiff(y_, y)
        y_[i] = 0
    return result


def test_np_softmax(batch_axis):
    """TODO."""
    N = batch_axis
    C = ng.make_axis(length=20)

    # set up some distributions
    u = np.empty((C.length, N.length))
    u = rng.uniform(0, 1, ng.make_axes([C, N]))
    u = u / sum(u, 0).reshape(1, N.length)

    # Put them in pre-softmax form
    x = np.log(u) + rng.uniform(-5000, 5000,
                                ng.make_axes([N])).reshape(1, N.length)

    s = np_softmax(x, 0)
    ng.testing.assert_allclose(s, u, atol=1e-6, rtol=1e-3)

    # Drop batch axis and test the derivative
    x0 = x[:, 0]

    def np_softmax_0(x):
        """
        TODO.

        Arguments:
          x: TODO

        Returns:

        """
        return np_softmax(x, 0)

    a = numeric_derivative(np_softmax_0, x0, .001)
    s = adiff_softmax(x0)
    ng.testing.assert_allclose(s, a, atol=1e-2, rtol=1e-2)


def np_cross_entropy_multi(y, t, axis=None):
    """
    TODO.

    Arguments:
      y: TODO
      t: TODO
      axis: TODO

    Returns:
      TODO
    """
    return -np.sum(np.log(y) * t, axis=axis)


@pytest.config.flex_disabled(reason="Results mismatch - too strict tolerance (rtol, atol)")
@pytest.config.argon_disabled  # TODO triage
def test_softmax(transformer_factory, input_tensor):
    """TODO."""
    p_x = input_tensor
    N = p_x.axes.batch_axes()[0]
    W = p_x.axes.sample_axes()[0]
    # set up some distributions
    u = rng.uniform(0, 1, p_x.axes)
    u = u / sum(u, 0).reshape(1, N.length)

    # Put them in pre-softmax form
    x = np.log(u) + rng.uniform(-5000, 5000,
                                ng.make_axes([N])).reshape(1, N.length)

    with ExecutorFactory() as ex:
        smax_w_fun = ex.executor(ng.softmax(p_x, normalization_axes=ng.make_axes([W])), p_x)
        smax_fun = ex.executor(ng.softmax(p_x), p_x)

        s = smax_w_fun(x)
        ng.testing.assert_allclose(s, u, atol=1e-6, rtol=1e-3)

        x = rng.uniform(-5000, 5000, p_x.axes)
        u = np_softmax(x, 0)
        s = smax_w_fun(x)
        ng.testing.assert_allclose(s, u, atol=1e-6, rtol=1e-3)

        # Test with softmax_axis default
        s = smax_fun(x)
        ng.testing.assert_allclose(s, u, atol=1e-6, rtol=1e-3)


def test_softmax2(transformer_factory, input_tensor):
    p_x = input_tensor
    x = rng.uniform(0, 1, p_x.axes)

    compare_f_at_x(ng.softmax(p_x), p_x, lambda x: np_softmax(x, 0), x, rtol=1e-5)


def test_softmax_deriv(transformer_factory, input_tensor):
    p_x = input_tensor
    x = rng.uniform(0, 1, p_x.axes)

    check_derivative(ng.softmax(p_x), p_x, 0.001, x, atol=1e-2, rtol=1e-2)


def test_softmax_rec(transformer_factory, recurrent_input_tensor):
    p_x = recurrent_input_tensor
    x = rng.uniform(0, 1, p_x.axes)

    compare_f_at_x(ng.softmax(p_x), p_x, lambda x: np_softmax(x, 0), x, rtol=1e-5)


def test_softmax_rec_deriv(transformer_factory, recurrent_input_tensor):
    p_x = recurrent_input_tensor
    x = rng.uniform(0, 1, p_x.axes)

    check_derivative(ng.softmax(p_x), p_x, 0.001, x, atol=1e-2, rtol=1e-2)


def test_cross_entropy_softmax(transformer_factory, input_tensor):
    p_x = input_tensor
    p_t = ng.placeholder(p_x.axes)

    cross_entropy_sm_x_t = ng.cross_entropy_multi(ng.softmax(p_x), p_t)

    x = rng.uniform(0, 1, p_x.axes)
    t = np_softmax(rng.uniform(0, 1, p_t.axes), 0)

    def f_np(x, t):
        return np_cross_entropy_multi(np_softmax(x, 0), t, axis=0)

    compare_f_at_x(cross_entropy_sm_x_t, [p_x, p_t], f_np, [x, t], rtol=1e-5)


def test_cross_entropy_softmax_deriv(transformer_factory, input_tensor):
    p_x = input_tensor
    p_t = ng.placeholder(p_x.axes)

    x = rng.uniform(0, 1, p_x.axes)
    t = np_softmax(rng.uniform(0, 1, p_t.axes), 0)

    check_derivative(
        ng.cross_entropy_multi(ng.softmax(p_x), p_t),
        p_x, 0.001, x,
        parameters=[p_t],
        parameter_values=[t],
        atol=1e-2, rtol=1e-2
    )


def test_cross_entropy_rec(transformer_factory, recurrent_input_tensor):
    p_x = recurrent_input_tensor
    p_t = ng.placeholder(p_x.axes)

    cross_entropy_sm_x_t = ng.cross_entropy_multi(ng.softmax(p_x), p_t)

    x = rng.uniform(0, 1, p_x.axes)
    t = np_softmax(rng.uniform(0, 1, p_t.axes), 0)

    def f_np(x, t):
        return np_cross_entropy_multi(np_softmax(x, 0), t, axis=0)

    compare_f_at_x(cross_entropy_sm_x_t, [p_x, p_t], f_np, [x, t], rtol=1e-5)


def test_cross_entropy_softmax_rec_deriv(transformer_factory, recurrent_input_tensor):
    p_x = recurrent_input_tensor
    p_t = ng.placeholder(p_x.axes)

    x = rng.uniform(0, 1, p_x.axes)
    t = np_softmax(rng.uniform(0, 1, p_t.axes), 0)

    check_derivative(
        ng.cross_entropy_multi(ng.softmax(p_x), p_t),
        p_x, 0.001, x,
        parameters=[p_t],
        parameter_values=[t],
        atol=1e-2, rtol=1e-2
    )


def test_cross_entropy_multi_unmatched_axes(input_tensor):
    """If y and t have different axes, an error should be thrown immediately"""
    y = input_tensor
    feature_axis, batch_axis = y.axes
    t = ng.placeholder([ng.make_axis(feature_axis.length), batch_axis])

    with pytest.raises(ng.UnmatchedAxesError):
        ng.cross_entropy_multi(y, t)


def test_cross_entropy_multi_axis_order(transformer_factory, input_tensor):
    """If y and t have different axis orders, it should give the same result"""
    y = input_tensor
    t1 = ng.placeholder(y.axes)

    # Reorder axes
    feature_axis, batch_axis = y.axes
    t2 = ng.placeholder(ng.make_axes([batch_axis, feature_axis]))

    # Set up numpy variables
    np_y = np.random.uniform(0, 1, y.axes.lengths)
    if feature_axis.length > batch_axis.length:
        np_t1 = np.eye(feature_axis.length)[:, :batch_axis.length]
    else:
        np_t1 = np.eye(batch_axis.length)[:feature_axis.length, :]
    np_t2 = np_t1.T

    with ExecutorFactory() as ex:
        f1 = ex.executor(ng.cross_entropy_multi(ng.softmax(y), t1), y, t1)
        f2 = ex.executor(ng.cross_entropy_multi(ng.softmax(y), t2), y, t2)

        out1 = f1(np_y, np_t1)
        out2 = f2(np_y, np_t2)
        ng.testing.assert_allclose(out1.ravel(), out2.ravel(), rtol=1e-5)


def test_sigmoid_deriv(transformer_factory, input_tensor):
    """TODO."""
    p_u = input_tensor
    u = rng.uniform(-3.0, 3.0, p_u.axes)

    val_u = ng.sigmoid(p_u)

    check_derivative(val_u, p_u, 0.001, u, atol=1e-2, rtol=1e-2)


def test_log_sigmoid_deriv(transformer_factory, input_tensor):
    """TODO."""
    p_u = input_tensor
    u = rng.uniform(-3.0, 3.0, p_u.axes)

    log_val_u = ng.log(ng.sigmoid(p_u))

    check_derivative(log_val_u, p_u, 0.001, u, atol=1e-2, rtol=1e-2)


def compare_f_at_x(f_be, x_be, f_np, x, **kwargs):
    """
    Compare op_graph implementation of a function with numpy implementation

    Arguments:
        f_be: op_graph function
        x_be: argument to op_graph
        f_np: numpy function
        x: value to pass in to both implementations of f
        kwargs: used to pass rtol/atol on to assert_allclose
    """
    # op_graph
    with ExecutorFactory() as ex:

        # if x_be and x are not tuples or lists, put them in lists with length 1
        if isinstance(x_be, (tuple, list)):
            assert len(x_be) == len(x)
        else:
            x_be = [x_be]
            x = [x]

        # numpy
        val_np = f_np(*x)

        val_be = ex.executor(f_be, *x_be)(*x)

        # compare numpy and op_graph
        ng.testing.assert_allclose(val_np, val_be, **kwargs)


def test_sigmoid_value(transformer_factory, input_tensor):
    """ check the output of sigmoid is the same as np """
    p_x = input_tensor
    x = rng.uniform(-3.0, 3.0, p_x.axes)

    compare_f_at_x(ng.sigmoid(p_x), p_x, lambda x: 1.0 / (1 + np.exp(-x)), x, rtol=1e-5)


def one_hot_comparison(hot_axes, axes, C):
    """
    TODO.

    Arguments:
      hot_axes: TODO
      axes: TODO
    """
    u = rng.random_integers(0, C.length - 1, axes, dtype=np.int8)
    u_p = ng.placeholder(axes, dtype=u.dtype)
    v = np.zeros(hot_axes.lengths, dtype=np.float32)
    udxiter = np.nditer(u, flags=['multi_index'])
    for uiter in udxiter:
        vindex = [int(uiter)]
        vindex.extend(udxiter.multi_index)
        v[tuple(vindex)] = 1

    with executor(ng.one_hot(u_p, axis=C), u_p) as ex:
        v_t = ex(u)
        ng.testing.assert_allclose(v_t, v)


def test_onehot(transformer_factory):
    """TODO."""
    C = ng.make_axis(length=4)
    H = ng.make_axis(length=32)
    W = ng.make_axis(length=32)
    N = ng.make_axis(length=128, name='N')

    one_hot_comparison(ng.make_axes([C, N]), ng.make_axes([N]), C)
    one_hot_comparison(ng.make_axes([C, W, H, N]), ng.make_axes([W, H, N]), C)


def test_clip(transformer_factory):
    H = ng.make_axis(length=5)
    W = ng.make_axis(length=4)
    axes = ng.make_axes([W, H])

    p_x = ng.placeholder(axes)
    x = (2 * rng.uniform(0, 1, axes) - 1) * 20
    clip_value = 10

    clip_func = ng.minimum(ng.maximum(p_x, -abs(clip_value)), abs(clip_value))

    # numpy results as expected results
    expected_result = np.clip(x, -abs(clip_value), abs(clip_value))

    with ExecutorFactory() as ex:
        costfunc = ex.executor(clip_func, p_x)
        result = costfunc(x)
        ng.testing.assert_allclose(result, expected_result)


def test_elementwise_fp16_in(transformer_factory):
    axes = ng.make_axes([ng.make_axis(length=2), ng.make_axis(length=2)])

    a = ng.constant(np.array([[1.0, 2.0], [4.0, 12.0]], dtype='float16'), axes,
                    dtype=np.dtype(np.float16))
    b = ng.constant(np.array([[1.0, 2.0], [6.0, 12.0]], dtype='float16'), axes,
                    dtype=np.dtype(np.float16))

    c = ng.multiply(a, b)

    with executor(c) as ex:
        result = ex()
        ng.testing.assert_allclose(result, [[1.0, 4.0], [24.0, 144.0]])


def test_elementwise_fp16_out(transformer_factory):
    axes = ng.make_axes([ng.make_axis(length=2), ng.make_axis(length=2)])

    a = ng.constant(np.array([[1.0, 2.0], [4.0, 12.0]], dtype='float32'), axes)
    b = ng.constant(np.array([[1.0, 2.0], [6.0, 12.0]], dtype='float32'), axes)

    c = ng.multiply(a, b, dtype=np.dtype(np.float16))

    with executor(c) as ex:
        result = ex()
        ng.testing.assert_allclose(result, [[1.0, 4.0], [24.0, 144.0]])


def test_argmax(transformer_factory):
    axes = ng.make_axes([ng.make_axis(length=2), ng.make_axis(length=8)])
    a = ng.placeholder(axes=axes)
    b = ng.argmax(a, out_axes=axes[0])

    with ExecutorFactory() as ex:
        func = ex.executor(b, a)
        baseline = func(np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                                  [8, 7, 6, 5, 4, 3, 2, 1]],
                                 dtype=np.float32))
        expected = np.array([7, 0])
        ng.testing.assert_allclose(baseline, expected)


@pytest.config.argon_disabled  # TODO triage
def test_fill_slice(transformer_factory):
    axes = ng.make_axes([ng.make_axis(length=2), ng.make_axis(length=8)])
    a = ng.placeholder(axes=axes)
    b = ng.sequential([ng.fill(a[:, 1], 0), ng.value_of(a)])

    with ExecutorFactory() as ex:
        func = ex.executor(b, a)
        baseline = func(np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                                  [8, 7, 6, 5, 4, 3, 2, 1]],
                                 dtype=np.float32))
        expected = np.array([[1, 0, 3, 4, 5, 6, 7, 8],
                             [8, 0, 6, 5, 4, 3, 2, 1]])
        ng.testing.assert_allclose(baseline, expected)


def test_empty_finalize():
    """Evaluating an empty CPUTransformer shouldn't raise any exceptions."""
    with ExecutorFactory() as ex:
        ex.transformer.initialize()


def test_tensor_derivative():
    """
    Ensure that a dTensor/dTensor fails if error tensor is not provided.
    """
    p = ng.placeholder(ng.make_axis(length=5))
    with pytest.raises(ValueError):
        ng.deriv(p, p)


@pytest.config.argon_disabled  # TODO triage
def test_mean(transformer_factory, input_tensor):
    inputs = input_tensor
    targets = ng.placeholder(inputs.axes)

    inp_stat = ng.mean(inputs, reduction_axes=inputs.axes.batch_axes())
    err = ng.sum(inp_stat - targets, out_axes=())
    with executor(err, inputs, targets) as comp_func:

        input_value = rng.uniform(-1, 1, inputs.axes)
        target_value = rng.uniform(-1, 1, targets.axes)
        ng_f_res = comp_func(input_value, target_value)

        np_f_res = np.sum(np.mean(input_value, axis=1, keepdims=True) - target_value)

        ng.testing.assert_allclose(np_f_res, ng_f_res, atol=1e-4, rtol=1e-4)


@pytest.config.argon_disabled  # TODO triage
def test_variance_wgrad(transformer_factory, input_tensor):
    inputs = input_tensor
    targets = ng.placeholder(inputs.axes)

    inp_stat = ng.variance(inputs, reduction_axes=inputs.axes.batch_axes())
    err = ng.sum(inp_stat - targets, out_axes=())
    d_inputs = ng.deriv(err, inputs)
    with executor([err, d_inputs], inputs, targets) as comp_func:

        input_value = rng.uniform(-0.1, 0.1, inputs.axes)
        target_value = rng.uniform(-0.1, 0.1, targets.axes)
        ng_f_res, ng_b_res = comp_func(input_value, target_value)

        np_f_res = np.sum(np.var(input_value, axis=1, keepdims=True) - target_value)

        ng.testing.assert_allclose(np_f_res, ng_f_res, atol=1e-4, rtol=1e-4)

        np_b_res = 2 * (input_value - np.mean(input_value, axis=1, keepdims=True))

        ng.testing.assert_allclose(np_b_res, ng_b_res, atol=1e-4, rtol=1e-4)


@pytest.config.argon_disabled  # TODO triage
def test_variance_sqrt_inverse(transformer_factory, input_tensor):
    inputs = input_tensor
    targets = ng.placeholder(inputs.axes)

    epsilon = 1e-3

    inp_stat = ng.reciprocal(
        ng.sqrt(
            ng.variance(inputs, reduction_axes=inputs.axes.batch_axes()) + epsilon
        )
    )
    err = ng.sum(inp_stat - targets, out_axes=())
    d_inputs = ng.deriv(err, inputs)
    with executor([err, d_inputs], inputs, targets) as comp_func:

        input_value = rng.uniform(-1, 1, inputs.axes)
        target_value = rng.uniform(-1, 1, targets.axes)
        ng_f_res, ng_b_res = comp_func(input_value, target_value)

        npv = np.var(input_value, axis=1, keepdims=True) + epsilon
        np_f_res = 1.0 / np.sqrt(npv)

        npv_delta = 2 * (input_value - np.mean(input_value, axis=1, keepdims=True))

        np_b_res = - 0.5 * np_f_res / npv * npv_delta

        np_f_res = np.sum(np_f_res - target_value)

        ng.testing.assert_allclose(np_f_res, ng_f_res, atol=1e-4, rtol=1e-4)
        ng.testing.assert_allclose(np_b_res, ng_b_res, atol=1e-4, rtol=1e-4)


def test_return_type(transformer_factory):
    x = ng.placeholder(())
    with ExecutorFactory() as ex:
        c0 = ex.executor(x, x)
        c1 = ex.executor([x], x)

        r0 = c0(1)
        assert r0 == 1

        r1 = c1(1)
        assert isinstance(r1, collections.Sequence)
        assert r1[0] == 1


def test_empty_computation(transformer_factory):
    with ExecutorFactory() as ex:
        computation = ex.executor(None)
        res = computation()
        assert not res


def test_wrong_placeholders(transformer_factory):
    x = ng.placeholder(())
    with ExecutorFactory() as ex:
        c = ex.executor(x, x)

        with pytest.raises(ValueError):
            c()

        with pytest.raises(ValueError):
            c(1, 2)

        assert c(1) == 1


@pytest.config.argon_disabled  # TODO triage
def test_broadcast_deriv_reorder(transformer_factory):
    H = ng.make_axis(2)
    W = ng.make_axis(3)

    x = ng.constant(np.random.rand(2, 3), axes=[H, W])
    x_broadcast = ng.broadcast(x, [W, H])
    x_sum = ng.sum(x_broadcast, out_axes=())
    dx = ng.deriv(x_sum, x)

    with ExecutorFactory() as ex:
        dx_fun = ex.executor(dx)
        ng.testing.assert_allclose(dx_fun(), np.ones((2, 3)))


@pytest.mark.xfail(pytest.config.getvalue("transformer") == "gpu",
                   reason="GPU problem with uint32 converting", strict=True)
def test_multiply_unit32_convertion(transformer_factory):
    x = ng.placeholder(axes=(), dtype=np.uint32())
    multiplier = 1
    ng_mul = 0.5 * x * 0.5

    with executor(ng_mul, x) as ex:
        ng_result = ex(multiplier)

    assert ng_result == 0.25
