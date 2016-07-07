import numpy as np

from neon.initializers import Uniform, Constant, Array

import geon.backends.graph.funs as be
import geon.backends.graph.pycudatransform as evaluation
import geon.backends.graph.axis as ax

"""
Test the initialization of be.Variable

"""
def test_init_variable():
    with be.bound_environment():

        ax.Y.length = 1
        hello = be.Variable(axes=(ax.Y,), init=Constant(4))

        enp = be.NumPyTransformer(results=[hello])

        result = enp.evaluate()[hello]
        print(result)

        assert(result == 4)
        print("pass variable initialization")

def test_init_1D_variable():
    with be.bound_environment():

        ax.Y.length = 2
        hello = be.Variable(axes=(ax.Y,), init=Constant(3))

        enp = be.NumPyTransformer(results=[hello])

        result = enp.evaluate()[hello]
        print(result)

        print("pass 1D variable initialization")

def test_init_2D_variable():
    with be.bound_environment():

        ax.M.length = 3
        ax.N.length = 2
        hello = be.Variable(axes=[ax.M, ax.N], init=Constant(5))

        enp = be.NumPyTransformer(results=[hello])

        result = enp.evaluate()[hello]
        print(result)

        print("pass 2D variable initialization")

def test_init_numpy_variable():
    with be.bound_environment():
        ax.Y.length = 10
        npvar = be.NumPyTensor(np.arange(ax.Y.length, dtype=np.float32), axes=[ax.Y])
        hello = be.Variable(axes=[ax.Y], init=Array(npvar))

        enp = be.NumPyTransformer(results=[hello])

        result = enp.evaluate()[hello]
        print(result)

        print("pass numpy variable initialization")

test_init_variable()
test_init_1D_variable()
test_init_2D_variable()
test_init_numpy_variable()
