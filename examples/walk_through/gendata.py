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
import numpy as np
import numbers


class MixtureGenerator(object):

    def __init__(self, pvals, shape, seed=0):
        if isinstance(shape, numbers.Integral):
            shape = (shape,)
        self.__rng = np.random.RandomState(seed)
        self.nclasses = len(pvals)
        self.shape = shape
        self.size = 1
        for s in shape:
            self.size = self.size * s
        self.As = self.__rng.uniform(-1, 1, (self.size, self.size, self.nclasses,))
        self.bs = self.__rng.uniform(-1, 1, (self.size, self.nclasses,))

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
        xs[...] = self.__rng.normal(0, 0.3, xs.shape)
        for i in range(ys.size):
            y = ys[i]
            x = xs[..., i].reshape(self.size)
            A = self.As[..., y]
            b = self.bs[..., y]
            x0 = np.dot(A, x) + b
            xs[..., i] = x0.reshape(self.shape)

    def make_mixture(self, N):
        return np.empty(self.shape + (N,)), np.empty((N,), dtype=int)

    def gen_data(self, batch_size, n_batches):
        XS = []
        YS = []
        for i in range(n_batches):
            xs, ys = self.make_mixture(batch_size)
            self.fill_mixture(xs, ys)
            XS.append(xs)
            YS.append(ys)
        return XS, YS
