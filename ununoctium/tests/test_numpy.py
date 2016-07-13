from geon.backends.graph.graphneon import *

env = be.Environment()


def execute(nodes):
    trans = be.NumPyTransformer(results=nodes)
    result = trans.evaluate()
    return (result[node] for node in nodes)


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
        rng = np.random.RandomState(seed=0)
        u = np.array(rng.uniform(-1.0, 1.0, ashape), dtype=np.float32)
        x.value = u
        cval, = execute([x])
        assert np.array_equal(cval, u)

        d = 2*x
        d2 = be.dot(x, x)
        x.value = aval
        cval, s = execute([d, d2])
        assert np.array_equal(cval, aval*2)
        assert s[()] == np.dot(aval, aval)

        x.value = u
        cval, s = execute([d, l2])
        u2 = u*2
        assert np.array_equal(cval, u2)
        assert s[()] == bp.dot(u, u)









