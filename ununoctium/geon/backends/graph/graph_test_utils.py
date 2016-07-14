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

