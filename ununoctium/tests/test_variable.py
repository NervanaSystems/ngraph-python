'''
Test the usage of be.Variable

'''

from neon.initializers import Constant

import geon.backends.graph.funs as be
import geon.backends.graph.axis as ax

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

def test_init_1D_variable_from_array():
    with be.bound_environment():

        ax.Y.length = 2
        hello = be.Variable(axes=(ax.Y,), init=Constant([3,3]))

        enp = be.NumPyTransformer(results=[hello])

        result = enp.evaluate()[hello]
        print(result)


test_init_variable()
test_init_1D_variable()
test_init_2D_variable()
test_init_1D_variable_from_array()