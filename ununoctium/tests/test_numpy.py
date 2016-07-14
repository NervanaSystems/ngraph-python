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
        ashape = arrayaxes.axes_shape(aaxes)
        asize = arrayaxes.axes_size(aaxes)
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
        ax.W.length = 10
        ax.H.length = 20
        axes = [ax.C, ax.W, ax.H]

        u = rng.uniform(-1000.0, 1000.0, axes)
        p_u = be.placeholder(axes=axes)
        p_u.value = u

        for npred, bered, red in [(np.sum, be.sum, 'sum'), (np.max, be.max, 'max'), (np.min, be.min, 'min')]:
            cured = npred(u, 0)  # WH
            wured = npred(u, 1)  # CH
            hured = npred(u, 2)  # CW

            cred, wred, hred = execute(
                [bered(p_u, reduction_axes=[ax.C]),
                 bered(p_u, reduction_axes=[ax.W]),
                 bered(p_u, reduction_axes=[ax.H])])

            assert np.array_equal(cured, cred), red
            assert np.array_equal(wured, wred), red
            assert np.array_equal(hured, hred), red

            cwured = npred(u, (0, 1))
            whured = npred(u, (1, 2))

            cwmax, whmax = execute(
                [bered(p_u, reduction_axes=[ax.C, ax.W]),
                 bered(p_u, reduction_axes=[ax.W, ax.H])]
            )

            assert np.array_equal(cwured, cwmax), red
            assert np.array_equal(whured, whmax), red


def np_softmax(x, axis):
    # Shape for broadcasts
    shape = list(x.shape)
    shape[axis] = 1

    exps = np.exp(x - np.max(x, axis).reshape(shape))
    return exps / np.sum(exps, axis).reshape(shape)


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

