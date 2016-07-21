from __future__ import print_function
import numpy as np

from neon.initializers import Uniform, Constant, Array

'''
Test the usage of transformer.allreduce

'''
import geon.backends.graph.funs as be
import geon.backends.graph.axis as ax


def test_allreduce():
    with be.bound_environment():
        ax.Y.length = 1
        hello = be.Variable(axes=(ax.Y,), init=Constant(4))

        enp = be.NumPyTransformer(results=[hello])

        result = enp.evaluate()[hello]
        print(result)

        assert (result == 4)
        print("pass variable initialization")



test_allreduce()

