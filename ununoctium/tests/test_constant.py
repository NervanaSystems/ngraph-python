'''
Test the usage of be.Constant

'''

import geon.backends.graph.funs as be
from neon.initializers import Constant

def test_constant_init():
    with be.bound_environment():
        a = be.Constant(5)

        enp = be.NumPyTransformer(results=[a])

        result = enp.evaluate()[a]
        print(result)

        assert(result == 5)
        print("pass constant initialization")


def test_constant_add():
    with be.bound_environment():
        a = be.Constant(1)
        b = be.Constant(2)
        c = a + b

        enp = be.NumPyTransformer(results=[c])

        result = enp.evaluate()[c]
        print(result)

        assert(result == 3)
        print("pass constant add")

def test_2D_constant_init():
    a = Constant([3, 5])
    print(a.val)

    with be.bound_environment():
        b = be.Constant([3,5])
        enp = be.NumPyTransformer(results=[b])
        result = enp.evaluate()[b]
        print(result)

test_constant_init()
test_constant_add()
test_2D_constant_init()