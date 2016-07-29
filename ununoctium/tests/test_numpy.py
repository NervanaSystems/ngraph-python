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

import geon.backends.graph.funs as be
import geon.backends.graph.axis as ax
from geon.backends.graph.arrayaxes import Axes
from geon.util.utils import RandomTensorGenerator, execute, transform_numeric_derivative
from geon.util.utils import transform_derivative, numeric_derivative

rng = RandomTensorGenerator(0, np.float32)


def test_constants():
    with be.bound_environment():
        # Simple evaluation of a scalar
        val = 5
        x = be.Constant(5)

        cval, = execute([x])
        assert cval.shape == ()
        assert cval[()] == val

        ax.W.length = 10
        ax.H.length = 20
        aaxes = Axes(ax.W, ax.H)
        ashape = aaxes.lengths
        asize = aaxes.size
        aval = np.arange(asize, dtype=np.float32).reshape(ashape)

        # Pass a NumPy array through as a constant
        x = be.NumPyTensor(aval, axes=aaxes)
        cval, = execute([x])
        assert np.array_equal(cval, aval)

        # Pass array through a placeholder
        x = be.placeholder(axes=[ax.W, ax.H])
        x.value = aval
        cval, = execute([x])
        assert np.array_equal(cval, aval)

        # Pass a different array though
        u = rng.uniform(-1.0, 1.0, aaxes)
        x.value = u
        cval, = execute([x])
        assert np.array_equal(cval, u)

        d = 2 * x
        d2 = be.dot(x, x)
        x.value = aval
        cval, s = execute([d, d2])
        assert np.array_equal(cval, aval * 2)
        assert s[()] == np.dot(aval.flatten(), aval.flatten())

        x.value = u
        cval, s = execute([d, d2])
        u2 = u * 2
        assert np.array_equal(cval, u2)
        assert s[()] == np.dot(u.flatten(), u.flatten())


def test_reduction():
    with be.bound_environment():
        ax.C.length = 4
        ax.W.length = 4
        ax.H.length = 4
        axes = Axes(ax.C, ax.W, ax.H)

        u = rng.uniform(-1.0, 1.0, axes)
        p_u = be.placeholder(axes=axes)
        p_u.value = u

        for npred, bered, red in [(np.sum, be.sum, 'sum'),
                                  (np.max, be.max, 'max'),
                                  (np.min, be.min, 'min')]:
            for reduction_axes in [[ax.C],
                                   [ax.W],
                                   [ax.H],
                                   [ax.C, ax.W],
                                   [ax.W, ax.H]]:
                dims = tuple(axes.index(axis) for axis in reduction_axes)
                npval = npred(u, dims)
                graph_reduce = bered(p_u, reduction_axes=reduction_axes)
                graph_val, = execute([graph_reduce])
                assert np.array_equal(
                    npval, graph_val), 'red:{red}, axes:{axes}'.format(
                    red=red, axes=reduction_axes)


def test_reduction_deriv():
    with be.bound_environment():
        delta = .001
        ax.C.length = 4
        ax.W.length = 10
        ax.H.length = 10
        axes = Axes(ax.C, ax.W, ax.H)

        u = rng.discrete_uniform(1.0, 2.0, 2 * delta, axes)

        p_u = be.placeholder(axes=axes)
        p_u.value = u

        # Need to test max/min differently since if two elements are extremums
        # and we modify one, the derivative will change.
        for npred, bered, red in [(np.sum, be.sum, 'sum')]:
            for reduction_axes in [[ax.C],
                                   [ax.W],
                                   [ax.H],
                                   [ax.C, ax.W],
                                   [ax.W, ax.H]]:
                # dims = tuple(axes.index(axis) for axis in reduction_axes)
                graph_reduce = bered(p_u, reduction_axes=reduction_axes)
                dgraph_val_num = transform_numeric_derivative(
                    graph_reduce, p_u, delta)
                dgraph_val = transform_derivative(graph_reduce, p_u)
                assert np.allclose(dgraph_val, dgraph_val_num, atol=1e-1,
                                   rtol=1e-1), 'red:{red}, axes:{axes}'.format(
                    red=red, axes=reduction_axes)


def test_reciprocal():
    with be.bound_environment():
        delta = .001
        ax.W.length = 20
        ax.N.length = 128
        axes = Axes(ax.W, ax.N)
        p_u = be.placeholder(axes=axes)
        u = rng.uniform(.1, 5.0, p_u.axes.value)
        p_u.value = u

        rec_u_np = np.reciprocal(u)
        rec_u = be.reciprocal(p_u)
        rec_u_graph, = execute([rec_u])
        assert np.allclose(rec_u_np, rec_u_graph)

        drec_u_num = transform_numeric_derivative(rec_u, p_u, delta)
        drec_u_graph = transform_derivative(rec_u, p_u)
        assert np.allclose(drec_u_graph, drec_u_num, atol=1e-2, rtol=1e-2)


def test_elementwise_ops_matched_args():
    with be.bound_environment():
        delta = .001
        ax.W.length = 20
        ax.H.length = 20
        ax.N.length = 128
        axes = Axes(ax.W, ax.H)

        for npop, beop, op in [(np.add, be.add, 'add'),
                               (np.subtract, be.subtract, 'sub'),
                               (np.multiply, be.multiply, 'multiply'),
                               (np.divide, be.divide, 'divide')]:
            # Matched sizes
            p_u = be.placeholder(axes=axes)
            p_v = be.placeholder(axes=axes)
            u = rng.uniform(-1.0, 1.0, p_u.axes.value)
            v = rng.uniform(1.0, 2.0, p_v.axes.value)
            uv_np = npop(u, v)
            p_u.value = u
            p_v.value = v
            top = beop(p_u, p_v)

            uv_t, = execute([top])
            assert np.allclose(uv_np, uv_t, atol=1e-4,
                               rtol=1e-4), 'op:{op}'.format(op=op)
            duvdunum = transform_numeric_derivative(top, p_u, delta)
            dudvdut = transform_derivative(top, p_u)
            assert np.allclose(duvdunum, dudvdut, atol=1e-4,
                               rtol=1e-4), 'op:{op}'.format(op=op)

            duvdvnum = transform_numeric_derivative(top, p_v, delta)
            dudvdvt = transform_derivative(top, p_v)
            assert np.allclose(duvdvnum, dudvdvt, atol=1e-3,
                               rtol=1e-3), 'op:{op}'.format(op=op)

        for npop, beop, op in [(np.exp, be.exp, 'exp'),
                               (np.log, be.log, 'log'),
                               (np.tanh, be.tanh, 'tanh')]:
            p_u = be.placeholder(axes=axes)
            u = rng.uniform(1.0, 2.0, p_u.axes.value)
            u_np = npop(u)
            p_u.value = u
            top = beop(p_u)

            u_t, = execute([top])
            assert np.allclose(u_np, u_t, atol=1e-4,
                               rtol=1e-4), 'op:{op}'.format(op=op)
            dudunum = transform_numeric_derivative(top, p_u, delta)
            dudut = transform_derivative(top, p_u)
            assert np.allclose(dudunum, dudut, atol=1e-3,
                               rtol=1e-3), 'op:{op}'.format(op=op)


def test_elementwise_ops_unmatched_args():
    with be.bound_environment():
        # delta = .001
        ax.W.length = 5
        ax.H.length = 5
        ax.N.length = 32
        sample_axes = [ax.W, ax.H]
        batch_axes = [ax.W, ax.H, ax.N]
        broadcast_dims = (ax.W.length, ax.H.length, 1)

        for npop, beop, op in [(np.add, be.add, 'add'),
                               (np.subtract, be.subtract, 'sub'),
                               (np.multiply, be.multiply, 'multiply'),
                               (np.divide, be.divide, 'divide')]:
            # Matched sizes
            p_u = be.placeholder(axes=sample_axes)
            p_v = be.placeholder(axes=batch_axes)
            u = rng.uniform(1.0, 2.0, p_u.axes.value)
            v = rng.uniform(1.0, 2.0, p_v.axes.value)

            p_u.value = u
            p_v.value = v

            # u op v
            uv_np = npop(u.reshape(broadcast_dims), v)
            top = beop(p_u, p_v)

            uv_t, = execute([top])
            assert np.allclose(uv_np, uv_t, atol=1e-4,
                               rtol=1e-4), 'op:{op}'.format(op=op)
            duvdunum = transform_numeric_derivative(top, p_u, .001)
            dudvdut = transform_derivative(top, p_u)
            assert np.allclose(duvdunum, dudvdut, atol=1e-3,
                               rtol=1e-3), 'op:{op}'.format(op=op)

            duvdvnum = transform_numeric_derivative(top, p_v, .001)
            dudvdvt = transform_derivative(top, p_v)
            assert np.allclose(duvdvnum, dudvdvt, atol=1e-3,
                               rtol=1e-3), 'op:{op}'.format(op=op)

            # v op u
            uv_np = npop(v, u.reshape(broadcast_dims))
            top = beop(p_v, p_u)

            uv_t, = execute([top])
            assert np.allclose(uv_np, uv_t, atol=1e-4,
                               rtol=1e-4), 'op:{op}'.format(op=op)
            duvdunum = transform_numeric_derivative(top, p_u, .001)
            dudvdut = transform_derivative(top, p_u)
            assert np.allclose(duvdunum, dudvdut, atol=1e-3,
                               rtol=1e-3), 'op:{op}'.format(op=op)

            duvdvnum = transform_numeric_derivative(top, p_v, .001)
            dudvdvt = transform_derivative(top, p_v)
            assert np.allclose(duvdvnum, dudvdvt, atol=1e-3,
                               rtol=1e-3), 'op:{op}'.format(op=op)


def np_softmax(x, axis):
    # Shape for broadcasts
    shape = list(x.shape)
    shape[axis] = 1

    exps = np.exp(x - np.max(x, axis).reshape(shape))
    return exps / np.sum(exps, axis).reshape(shape)


def cross_entropy_binary_logistic(x, t):
    y = 1.0 / (1.0 + np.exp(-x))
    return -(np.log(y) * t + np.log(1 - y) * (1 - t))


def cross_entropy_binary_logistic_shortcut(x, t):
    y = 1.0 / (1.0 + np.exp(-x))
    return (1.0 - t) * x - np.log(y)


def test_cross_entropy_binary_logistic_shortcut():
    with be.bound_environment():
        ax.W.length = 20
        ax.N.length = 128
        axes = Axes(ax.W, ax.N)
        p_u = be.placeholder(axes=axes)
        u = rng.uniform(-3.0, 3.0, p_u.axes.value)
        p_u.value = u
        p_v = be.placeholder(axes=axes)
        v = np_softmax(rng.uniform(-3.0, 3.0, p_u.axes.value), 0)
        p_v.value = v

        cel = cross_entropy_binary_logistic(u, v)
        cel_shortcut = cross_entropy_binary_logistic_shortcut(u, v)
        assert np.allclose(cel, cel_shortcut)

        cel_graph, = execute([be.cross_entropy_binary_inner(be.sig(p_u), p_v)])
        assert np.allclose(cel, cel_graph)


def test_cross_entropy_binary():
    with be.bound_environment():
        delta = .001
        ax.W.length = 20
        ax.N.length = 128
        axes = Axes(ax.W, ax.N)
        p_u = be.placeholder(axes=axes)
        u = rng.uniform(-3.0, 3.0, p_u.axes.value)
        p_u.value = u
        p_v = be.placeholder(axes=axes)
        v = rng.uniform(-3.0, 3.0, p_u.axes.value)
        p_v.value = v

        y = be.sig(p_u)
        t = be.softmax(p_v)
        val_u = be.cross_entropy_binary_inner(y, t)

        dval_u_num = transform_numeric_derivative(val_u, p_u, delta)
        dval_u_graph = transform_derivative(val_u, p_u)
        assert np.allclose(dval_u_graph, dval_u_num, atol=1e-2, rtol=1e-2)


def adiff_softmax(x):
    """
    The version of the diff we use in autodiff, without batch axis.
    :param x:
    :return:
    """

    def softmax_adiff(y_, y):
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


def test_np_softmax():
    with be.bound_environment():
        ax.N.length = 128
        ax.C.length = 20

        # set up some distributions
        u = np.empty((ax.C.length, ax.N.length))
        u = rng.uniform(0, 1, Axes(ax.C, ax.N))
        u = u / sum(u, 0).reshape(1, ax.N.length)

        # Put them in pre-softmax form
        x = np.log(u) + rng.uniform(-5000, 5000,
                                    Axes(ax.N)).reshape(1, ax.N.length)

        s = np_softmax(x, 0)
        assert np.allclose(s, u, atol=1e-6, rtol=1e-3)

        # Drop batch axis and test the derivative
        x0 = x[:, 0]

        def np_softmax_0(x):
            return np_softmax(x, 0)

        a = numeric_derivative(np_softmax_0, x0, .001)
        s = adiff_softmax(x0)
        assert np.allclose(s, a, atol=1e-2, rtol=1e-2)


def np_cross_entropy_multi(y, t, axis=None):
    return -np.sum(np.log(y) * t, axis=axis)


def test_softmax():
    with be.bound_environment():
        ax.W.length = 128
        ax.N.length = 10
        be.set_batch_axes([ax.N])
        axes = [ax.W, ax.N]

        # set up some distributions
        u = rng.uniform(0, 1, Axes(ax.W, ax.N))
        u = u / sum(u, 0).reshape(1, ax.N.length)

        # Put them in pre-softmax form
        x = np.log(u) + rng.uniform(-5000, 5000,
                                    Axes(ax.N)).reshape(1, ax.N.length)
        p_x = be.placeholder(axes=axes)
        p_x.value = x

        s, = execute([be.softmax(p_x, softmax_axes=Axes(ax.W))])
        assert np.allclose(s, u, atol=1e-6, rtol=1e-3)

        x = rng.uniform(-5000, 5000, Axes(ax.W, ax.N))
        p_x.value = x
        u = np_softmax(x, 0)
        s, = execute([be.softmax(p_x, softmax_axes=Axes(ax.W))])
        assert np.allclose(s, u, atol=1e-6, rtol=1e-3)

        # Test with softmax_axis default
        s, = execute([be.softmax(p_x)])
        assert np.allclose(s, u, atol=1e-6, rtol=1e-3)

        # Now try the derivative
        axes = Axes(ax.W, ax.N)
        ax.W.length = 3
        ax.N.length = 10

        x = rng.uniform(0, 1, axes)
        p_x = be.placeholder(axes=axes)
        p_x.value = x

        sx = be.softmax(p_x)
        sxval, = execute([sx])

        # npadiff = adiff_softmax(x)
        ndsx = transform_numeric_derivative(sx, p_x, .001)

        # assert np.allclose(npadiff, ndsx, atol=1e-2, rtol=1e-2)

        tdsx = transform_derivative(sx, p_x)
        assert np.allclose(ndsx, tdsx, atol=1e-2, rtol=1e-2)

        # Now try cross-entropy
        t = np_softmax(rng.uniform(0, 1, axes), 0)
        p_t = be.placeholder(axes=axes)
        p_t.value = t

        ce = be.cross_entropy_multi(sx, p_t)
        npsm = np_softmax(x, 0)
        nce = np_cross_entropy_multi(npsm, t, axis=0)
        tce, tsm = execute([ce, sx])
        assert np.allclose(nce, tce)

        def np_all(x):
            return np_cross_entropy_multi(np_softmax(x, 0), t, axis=0)

#       npdce = numeric_derivative(np_all, x, .001)

        ndce = transform_numeric_derivative(ce, p_x, .001)
        tdce = transform_derivative(ce, p_x)
        assert np.allclose(ndce, tdce, atol=1e-2, rtol=1e-2)


def test_sigmoid():
    with be.bound_environment():
        delta = .001
        ax.W.length = 20
        ax.N.length = 128
        axes = Axes(ax.W, ax.N)
        p_u = be.placeholder(axes=axes)
        u = rng.uniform(-3.0, 3.0, p_u.axes.value)
        p_u.value = u

        val_u_np = 1.0 / (1 + np.exp(-u))
        val_u = be.sig(p_u)
        val_u_graph, = execute([val_u])
        assert np.allclose(val_u_np, val_u_graph)

        dval_u_num = transform_numeric_derivative(val_u, p_u, delta)
        dval_u_graph = transform_derivative(val_u, p_u)
        assert np.allclose(dval_u_graph, dval_u_num, atol=1e-2, rtol=1e-2)

        val_u = be.log(val_u)
        dval_u_num = transform_numeric_derivative(val_u, p_u, delta)
        dval_u_graph = transform_derivative(val_u, p_u)
        assert np.allclose(dval_u_graph, dval_u_num, atol=1e-2, rtol=1e-2)


def one_hot_comparison(hot_axes, axes):
    u = rng.random_integers(0, ax.C.length - 1, axes, dtype=np.int8)
    u_p = be.placeholder(axes=axes, dtype=u.dtype)
    u_p.value = u
    v = np.zeros(hot_axes.lengths, dtype=np.float32)
    udxiter = np.nditer(u, flags=['multi_index'])
    for uiter in udxiter:
        vindex = [int(uiter)]
        vindex.extend(udxiter.multi_index)
        v[tuple(vindex)] = 1

    v_t, = execute([be.onehot(u_p, axis=ax.C)])
    assert np.array_equal(v_t, v)


def test_onehot():
    with be.bound_environment():
        ax.C.length = 4
        ax.W.length = 32
        ax.H.length = 32
        ax.N.length = 128
        be.set_batch_axes([ax.N])

        one_hot_comparison(Axes(ax.C, ax.N), Axes(ax.N))
        one_hot_comparison(Axes(ax.C, ax.W, ax.H, ax.N), Axes(ax.W, ax.H, ax.N))
