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
from builtins import object

import math
import numpy as np

from geon.frontends.neon import *  # noqa


class RandomTensorGenerator(object):

    def __init__(self, seed=0, dtype=np.float32):
        self.dtype = dtype
        self.reset(seed)

    def reset(self, seed=0):
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)

    def uniform(self, low, high, axes):
        return np.array(
            self.rng.uniform(
                low,
                high,
                Axes(axes).lengths),
            dtype=self.dtype)

    def discrete_uniform(self, low, high, quantum, axes):
        n = math.floor((high - low) / quantum)
        result = np.array(self.rng.random_integers(
            0, n, Axes(axes).lengths), dtype=self.dtype)
        np.multiply(result, quantum, result)
        np.add(result, low, result)
        return result

    def random_integers(self, low, high, axes, dtype=np.int8):
        return self.rng.random_integers(low, high, Axes(axes).lengths).astype(dtype)


def in_bound_environment(f):
    def wrapper(*args, **kargs):
        with be.bound_environment():
            return f(*args, **kargs)

    return wrapper


def with_error_settings(**new_settings):
    def decorator(f):
        def wrapper(*args, **kargs):
            old_settings = np.geterr()

            np.seterr(**new_settings)
            ret = f(*args, **kargs)

            np.seterr(**old_settings)

            return ret

        return wrapper

    return decorator


def raise_all_numpy_errors(f):
    settings = {k: 'raise' for k in ['divide', 'over', 'under', 'invalid']}
    return with_error_settings(**settings)(f)


def execute(nodes):
    """
    Evaluates nodes and returns their values.

    All necessary placeholders should already have values.
    :param nodes: list of nodes to evaluate.
    :return: list of values of the nodes after evaluation.
    """
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


def numeric_derivative(f, x, dx):
    """
    Computer df/dx at x numerically.
    Do not use for non-continuous derivatives such as min/max.  If there is a tie at the
    extremum, only one value will change and the computed derivative will be very wrong.

    Would be useful to have a batch axis some time.

    :param f: Tensor function.
    :param x: Derivative position.
    :param dx: scalar dx change in each dimension
    :return: Derivative, with f(x), x indexing, i.e. if f is 2x4 and x is 3x7, result is 2x4x3x7.
    """
    xshape = shape(x)
    # Copy because we always compute into the same place
    y = np.copy(f(x))
    fshape = shape(y)
    dshape = list(fshape)
    dshape.extend(xshape)
    d = np.zeros(shape=dshape, dtype=np.float32)
    dindex = [slice(None) for _ in fshape]
    dindex.extend((0 for _ in xshape))

    idxiter = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    for xiter in idxiter:
        old_x = np.float32(xiter)
        xiter[...] = old_x + dx
        dy = f(x) - y
        dindex[len(fshape):] = idxiter.multi_index
        d[tuple(dindex)] = (dy / dx)
        xiter[...] = old_x
    return d


def transform_numeric_derivative(f, p_x, dx):
    trans = be.NumPyTransformer(results=[f])

    def trans_softmax(x):
        p_x.value = x
        result = trans.evaluate()
        return result[f]

    return numeric_derivative(trans_softmax, p_x.value, dx)


def transform_derivative(f, px):
    """
    Full derivative of f wrt placeholder px
    :param f:
    :param px:
    :return:
    """
    x = px.value
    fshape = f.axes.lengths
    xshape = px.axes.lengths

    dfdx = be.deriv(f, px)

    trans = be.NumPyTransformer(results=[dfdx])
    if len(fshape) is 0:
        return trans.evaluate()[dfdx]
    else:
        dfdxshape = list(fshape)
        dfdxshape.extend(xshape)
        npdfdx = np.empty(dfdxshape, dtype=x.dtype)

        dindex = [0 for _ in fshape]
        dindex.extend([slice(None) for _ in xshape])

        adjoint = np.zeros(fshape, dtype=x.dtype)
        f.initial_adjoint.value = adjoint

        idxiter = np.nditer(
            adjoint, flags=['multi_index'], op_flags=['readwrite'])
        for dfdxiter in idxiter:
            dfdxiter[...] = 1
            df = trans.evaluate()[dfdx]
            dindex[0:len(fshape)] = idxiter.multi_index
            npdfdx[tuple(dindex)] = df
            dfdxiter[...] = 0

        return npdfdx
