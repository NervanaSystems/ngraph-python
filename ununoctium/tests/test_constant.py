'''
Test the usage of be.Constant

'''

import geon.backends.graph.funs as be
import geon.backends.graph.axis as ax
import numpy as np
from neon.initializers import Constant
from geon.backends.graph.environment import Environment

def test_constant_init():
    with be.bound_environment():
        a = be.Constant(5)

        enp = be.NumPyTransformer(results=[a])

        result = enp.evaluate()[a]
        print(result)

        assert (result == 5)
        print("pass constant initialization")


def test_constant_add():
    with be.bound_environment():
        a = be.Constant(1)
        b = be.Constant(2)
        c = a + b

        enp = be.NumPyTransformer(results=[c])

        result = enp.evaluate()[c]
        print(result)

        assert (result == 3)
        print("pass constant add")


def test_1D_constant_init():
    with be.bound_environment():
        ax.Y.length = 2
        a = be.NumPyTensor(np.array([3, 5], dtype=np.float32), axes=[ax.Y])
        b = be.NumPyTensor(np.array([3, 5], dtype=np.float32), axes=[ax.Y])
        c = a + b
        enp = be.NumPyTransformer(results=[c])
        result = enp.evaluate()
        print(result[c])

def test_2D_constant_init():
    with be.bound_environment():
        ax.M.length = 2
        ax.N.length = 2
        a = be.NumPyTensor(np.array([[1, 2],[3, 4]], dtype=np.float32), axes=[ax.M, ax.N])
        b = be.NumPyTensor(np.array([[1, 2],[3, 4]], dtype=np.float32), axes=[ax.M, ax.N])
        c = a + b
        enp = be.NumPyTransformer(results=[c])
        result = enp.evaluate()
        print(result[c])

def test_2D_constant_dot():
    with be.bound_environment():
        ax.M.length = 1
        ax.N.length = 3
        a = be.NumPyTensor(np.array([[1, 2, 3]], dtype=np.float32), axes=[ax.M, ax.N])

        ax.N.length = 3
        ax.Y.length = 2
        b = be.NumPyTensor(np.array([[1,2], [2,3], [3,4]], dtype=np.float32), axes=[ax.N, ax.Y])
        c = be.dot(a, b)
        enp = be.NumPyTransformer(results=[c])
        result = enp.evaluate()
        print(result[c])

def test_constant_multiply():
    with be.bound_environment():
        a = be.Constant([2, 4])
        b = be.Constant(2)
        c = be.multiply(a, b)
        enp = be.NumPyTransformer(results=[c])
        result = enp.evaluate()
        print(result[c])

# test_constant_init()
# test_constant_add()
# test_1D_constant_init()
# test_2D_constant_init()
test_constant_multiply()