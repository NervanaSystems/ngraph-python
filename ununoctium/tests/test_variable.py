import numpy as np

from neon.initializers import Uniform, Constant, Array

'''
Test the usage of be.Variable

'''
import geon.backends.graph.funs as be
import geon.backends.graph.axis as ax


def test_init_variable():
    with be.bound_environment():
        ax.Y.length = 1
        hello = be.Variable(axes=(ax.Y,), init=Constant(4))

        enp = be.NumPyTransformer(results=[hello])

        result = enp.evaluate()[hello]
        print(result)

        assert (result == 4)
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


def test_init_1D_variable_from_numpy_array():
    with be.bound_environment():
        ax.Y.length = 10
        npvar = be.NumPyTensor(np.arange(ax.Y.length, dtype=np.float32), axes=[ax.Y])
        hello = be.Variable(axes=[ax.Y], init=Array(npvar))

        enp = be.NumPyTransformer(results=[hello])

        result = enp.evaluate()[hello]
        print(result)

        print("pass numpy variable initialization")

def test_assign_1D_variable_with_numpy_tensor():
    with be.bound_environment():
        ax.Y.length = 10
        hello = be.Variable(axes=[ax.Y])
        npvar = be.NumPyTensor(np.arange(ax.Y.length, dtype=np.float32), axes=[ax.Y])

        hello = npvar
        enp = be.NumPyTransformer(results=[hello])
        result = enp.evaluate()[hello]
        print(result)

        print("pass 1D numpy tensor assignment")

def test_assign_2D_variable_with_numpy_tensor():
    with be.bound_environment():
        ax.M.length = 3
        ax.N.length = 2

        var = be.Variable(axes=[ax.M, ax.N])
        npvar = be.NumPyTensor(np.array([[1,1],[2,2],[3,3]]), axes=[ax.M, ax.N])

        var = npvar
        enp = be.NumPyTransformer(results=[var])
        result = enp.evaluate()[var]
        print(result)

        print("pass 2D numpy tensor assignment")

test_init_variable()
test_init_1D_variable()
test_init_2D_variable()
test_init_1D_variable_from_numpy_array()

test_assign_1D_variable_with_numpy_tensor()
test_assign_2D_variable_with_numpy_tensor()

