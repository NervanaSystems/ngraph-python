from geon.backends.graph.graph_test_utils import *

env = be.Environment()
rng = RandomTensorGenerator(0, np.float32)


def test_constants():
    with be.bound_environment(env):
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
    with be.bound_environment(env):
        ax.C.length = 4
        ax.W.length = 10
        ax.H.length = 20
        axes = [ax.C, ax.W, ax.H]

        u = rng.uniform(-1000.0, 1000.0, axes)
        p_u = be.placeholder(axes=axes)
        p_u.value = u

        for npred, bered, red in [(np.sum, be.sum, 'sum'), (np.max, be.max, 'max'), (np.min, be.min, 'min')]:

            cured = npred(u, 0) # WH
            wured = npred(u, 1) # CH
            hured = npred(u, 2) # CW

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
