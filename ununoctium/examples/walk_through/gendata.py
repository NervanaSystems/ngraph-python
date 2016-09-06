import numpy as np


class MixtureGenerator(object):

    def __init__(self, pvals, d, seed=0):
        self.__rng = np.random.RandomState(seed)
        self.nclasses = len(pvals)
        self.d = d
        self.As = self.__rng.uniform(-1, 1, (d, d, self.nclasses))
        self.bs = self.__rng.uniform(-1, 1, (d, self.nclasses))

        self.accum = []
        s = 0
        for pval in pvals:
            s = s + pval
            self.accum.append(s)
        self.accum[-1] = 2

    def multichoose(self):
        x = self.__rng.uniform(0, 1)
        for i, aval in enumerate(self.accum):
            if x < aval:
                return i

    def multinomial(self, ys):
        """
        Initialize y with multinomial values distributed per pvals.

        Arguments:
            ys: 1-d tensor.
        """
        i = 0
        for i in range(ys.size):
            ys[i] = self.multichoose()

    def fill_mixture(self, xs, ys):
        self.multinomial(ys)
        xs[...] = self.__rng.normal(0, 1, xs.shape)
        for i in range(ys.size):
            y = ys[i]
            x = xs[..., i]
            A = self.As[..., y]
            b = self.bs[..., y]
            x0 = np.dot(A, x) + b
            xs[..., i] = x0

    def make_mixture(self, N):
        return np.empty((self.d, N)), np.empty((N,), dtype=int)
