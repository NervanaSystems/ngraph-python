import numpy as np

from geon.backends.graph.arrayaxes import axes_shape
from geon.backends.graph.graphneon import *
import geon.backends.graph.arrayaxes as arrayaxes


class RandomTensorGenerator(object):
    def __init__(self, seed=0, dtype=np.float32):
        self.dtype = dtype
        self.reset(seed)

    def reset(self, seed=0):
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)

    def uniform(self, low, high, axes):
        return np.array(self.rng.uniform(low, high, axes_shape(axes)), dtype=self.dtype)


def execute(nodes):
    trans = be.NumPyTransformer(results=nodes)
    result = trans.evaluate()
    return (result[node] for node in nodes)


def shape(x):
    """
    Shape of a tensor/scalar
    :param x:
    :return:
    """
    if isinstance(x, np.ndarray):
        return x.shape
    else:
        return ()


def numeric_derivative(f, x, delta):
    """
    Computer df/dx at x numerically.

    Would be useful to have a batch axis some time.

    :param f: Tensor function.
    :param x: Derivative position.
    :param delta: scalar dx change in each dimension
    :return: Derivative, with f(x), x indexing, i.e. if f is 2x4 and x is 3x7, result is 2x4x3x7.
    """
    xshape = shape(x)
    # Copy because we always compute into the same place
    fx = np.copy(f(x))
    fshape = shape(fx)
    dshape = list(fshape)
    dshape.extend(xshape)
    d = np.zeros(shape=dshape, dtype=np.float32)
    dindex = [slice(None) for _ in fshape]
    dindex.extend((0 for _ in xshape))

    idxiter = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    for xiter in idxiter:
        old_x = np.float32(xiter)
        xiter[...] = old_x + delta
        fdx = f(x) - fx
        xindex = idxiter.multi_index
        dindex[len(fshape):] = xindex
        d[tuple(dindex)] = (fdx / delta)
        xiter[...] = old_x

    return d


def transform_numeric_derivative(f, p_x, delta):
    trans = be.NumPyTransformer(results=[f])

    def trans_softmax(x):
        p_x.value = x
        result = trans.evaluate()
        return result[f]

    return numeric_derivative(trans_softmax, p_x.value, delta)


def transform_derivative(f, px):
    """
    Full derivative of f wrt placeholder px
    :param f:
    :param px:
    :return:
    """
    x = px.value
    fshape = axes_shape(f.axes.value)
    xshape = axes_shape(px.axes.value)
    dfdxshape = list(fshape)
    dfdxshape.extend(xshape)
    npdfdx = np.empty(dfdxshape, dtype=x.dtype)

    dindex = [slice(None) for _ in fshape]
    dindex.extend((0 for _ in xshape))

    adjoint = np.zeros(fshape, dtype=x.dtype)
    padjoint = be.placeholder(axes=f.axes)
    padjoint.value = adjoint

    dfdx = be.deriv(f, px, padjoint)

    trans = be.NumPyTransformer(results=[dfdx])

    idxiter = np.nditer(adjoint, flags=['multi_index'], op_flags=['readwrite'])
    for xiter in idxiter:
        xiter[...] = 1
        df = trans.evaluate()[dfdx]
        xindex = idxiter.multi_index
        dindex[len(fshape):] = xindex
        npdfdx[tuple(dindex)] = df
        xiter[...] = 0

    return npdfdx
