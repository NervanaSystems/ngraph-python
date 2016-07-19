from __future__ import division
from builtins import range
from geon.backends.graph.graph_test_utils import *

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
        aaxes = [ax.W, ax.H]
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
        axes = [ax.C, ax.W, ax.H]

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
                assert np.array_equal(npval, graph_val), 'red:{red}, axes:{axes}'.format(red=red, axes=reduction_axes)


def test_reduction_deriv():
    with be.bound_environment():
        delta = .001
        ax.C.length = 4
        ax.W.length = 10
        ax.H.length = 10
        axes = [ax.C, ax.W, ax.H]

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
                dims = tuple(axes.index(axis) for axis in reduction_axes)
                graph_reduce = bered(p_u, reduction_axes=reduction_axes)
                dgraph_val_num = transform_numeric_derivative(graph_reduce, p_u, delta)
                dgraph_val = transform_derivative(graph_reduce, p_u)
                assert np.allclose(dgraph_val, dgraph_val_num, atol=1e-1, rtol=1e-1), 'red:{red}, axes:{axes}'.format(
                    red=red, axes=reduction_axes)


def test_elementwise_ops_matched_args():
    with be.bound_environment():
        delta = .001
        ax.W.length = 20
        ax.H.length = 20
        ax.N.length = 128
        axes = [ax.W, ax.H]

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
            assert np.allclose(uv_np, uv_t, atol=1e-4, rtol=1e-4), 'op:{op}'.format(op=op)
            duvdunum = transform_numeric_derivative(top, p_u, .001)
            dudvdut = transform_derivative(top, p_u)
            assert np.allclose(duvdunum, dudvdut, atol=1e-4, rtol=1e-4), 'op:{op}'.format(op=op)

            duvdvnum = transform_numeric_derivative(top, p_v, .001)
            dudvdvt = transform_derivative(top, p_v)
            assert np.allclose(duvdvnum, dudvdvt, atol=1e-3, rtol=1e-3), 'op:{op}'.format(op=op)

        for npop, beop, op in [(np.exp, be.exp, 'exp'),
                               (np.log, be.log, 'log'),
                               (np.tanh, be.tanh, 'tanh')]:
            p_u = be.placeholder(axes=axes)
            u = rng.uniform(1.0, 2.0, p_u.axes.value)
            u_np = npop(u)
            p_u.value = u
            top = beop(p_u)

            u_t, = execute([top])
            assert np.allclose(u_np, u_t, atol=1e-4, rtol=1e-4), 'op:{op}'.format(op=op)
            dudunum = transform_numeric_derivative(top, p_u, .001)
            dudut = transform_derivative(top, p_u)
            assert np.allclose(dudunum, dudut, atol=1e-3, rtol=1e-3), 'op:{op}'.format(op=op)


def test_elementwise_ops_unmatched_args():
    with be.bound_environment():
        delta = .001
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
            assert np.allclose(uv_np, uv_t, atol=1e-4, rtol=1e-4), 'op:{op}'.format(op=op)
            duvdunum = transform_numeric_derivative(top, p_u, .001)
            dudvdut = transform_derivative(top, p_u)
            assert np.allclose(duvdunum, dudvdut, atol=1e-3, rtol=1e-3), 'op:{op}'.format(op=op)

            duvdvnum = transform_numeric_derivative(top, p_v, .001)
            dudvdvt = transform_derivative(top, p_v)
            assert np.allclose(duvdvnum, dudvdvt, atol=1e-3, rtol=1e-3), 'op:{op}'.format(op=op)

            # v op u
            uv_np = npop(v, u.reshape(broadcast_dims))
            top = beop(p_v, p_u)

            uv_t, = execute([top])
            assert np.allclose(uv_np, uv_t, atol=1e-4, rtol=1e-4), 'op:{op}'.format(op=op)
            duvdunum = transform_numeric_derivative(top, p_u, .001)
            dudvdut = transform_derivative(top, p_u)
            assert np.allclose(duvdunum, dudvdut, atol=1e-3, rtol=1e-3), 'op:{op}'.format(op=op)

            duvdvnum = transform_numeric_derivative(top, p_v, .001)
            dudvdvt = transform_derivative(top, p_v)
            assert np.allclose(duvdvnum, dudvdvt, atol=1e-3, rtol=1e-3), 'op:{op}'.format(op=op)


def np_softmax(x, axis):
    # Shape for broadcasts
    shape = list(x.shape)
    shape[axis] = 1

    exps = np.exp(x - np.max(x, axis).reshape(shape))
    return exps / np.sum(exps, axis).reshape(shape)


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
        u = rng.uniform(0, 1, [ax.C, ax.N])
        u = u / sum(u, 0).reshape(1, ax.N.length)

        # Put them in pre-softmax form
        x = np.log(u) + rng.uniform(-5000, 5000, [ax.N]).reshape(1, ax.N.length)

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
        u = rng.uniform(0, 1, [ax.W, ax.N])
        u = u / sum(u, 0).reshape(1, ax.N.length)

        # Put them in pre-softmax form
        x = np.log(u) + rng.uniform(-5000, 5000, [ax.N]).reshape(1, ax.N.length)
        p_x = be.placeholder(axes=axes)
        p_x.value = x

        s, = execute([be.softmax(p_x, softmax_axes=[ax.W])])
        assert np.allclose(s, u, atol=1e-6, rtol=1e-3)

        x = rng.uniform(-5000, 5000, [ax.W, ax.N])
        p_x.value = x
        u = np_softmax(x, 0)
        s, = execute([be.softmax(p_x, softmax_axes=[ax.W])])
        assert np.allclose(s, u, atol=1e-6, rtol=1e-3)

        # Test with softmax_axis default
        s, = execute([be.softmax(p_x)])
        assert np.allclose(s, u, atol=1e-6, rtol=1e-3)

        # Now try the derivative
        axes = [ax.W, ax.N]
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

        npdce = numeric_derivative(np_all, x, .001)

        ndce = transform_numeric_derivative(ce, p_x, .001)
        tdce = transform_derivative(ce, p_x)
        assert np.allclose(ndce, tdce, atol=1e-2, rtol=1e-2)
