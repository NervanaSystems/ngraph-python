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
import math
import numpy as np
import ngraph as ng


class RandomTensorGenerator(object):
    """
    Generate various pseudo-random values from a seed.

    Arguments:
        seed: The seed for the random number generator.
        dtype: The type of the generated values.
    """

    def __init__(self, seed=0, dtype=np.float32):
        self.dtype = dtype
        self.seed = 0
        self.reset(seed)

    def reset(self, seed=None):
        """
        Restart generation from the seed.

        Arguments:
            seed: If supplied, a new seed for generation, otherwise the original seed.
        """
        if seed is not None:
            self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)

    def uniform(self, low, high, axes, dtype=None):
        """
        Returns a tensor initialized with a uniform distribution from low to high with axes.

        Arguments:
            low: The lower limit of the distribution.
            high: The upper limit of the distribution.
            axes: The axes of the tensor.
            dtype: If supplied, the type of the values.

        Returns:
            The initialized tensor.

        """
        if dtype is None:
            dtype = self.dtype

        return np.array(
            self.rng.uniform(
                low,
                high,
                ng.make_axes(axes).lengths),
            dtype=dtype)

    def discrete_uniform(self, low, high, quantum, axes, dtype=None):
        """
        Returns a tensor initialized with a discrete uniform distribution.

        Arguments:
            low: The lower limit of the values.
            high: The upper limit of the values.
            quantum: Distance between values.
            axes: The axes of the tensor.

        Returns:
            The tensor.

        """
        if dtype is None:
            dtype = self.dtype

        n = math.floor((high - low) / quantum)
        result = np.array(self.rng.random_integers(
            0, n, ng.make_axes(axes).lengths), dtype=dtype)
        np.multiply(result, quantum, result)
        np.add(result, low, result)
        return result

    def random_integers(self, low, high, axes, dtype=np.int8):
        """
        Returns a tensor initialized with random integers.

        Arguments:
            low: The lower limit of values.
            high: the upper limit of values.
            axes: The axes of the tensors.
            dtype: The dtype of the values.

        Returns:
            The tensor.
        """
        return self.rng.random_integers(low, high, ng.make_axes(axes).lengths).astype(dtype)

    def normal(self, loc, scale, axes, dtype=None):
        """
        Returns a tensor initialized with a normal distribution with mean loc and std scale

        Arguments:
            loc: Mean of the distribution
            scale: Standard deviation of the distribution
            axes: The axes of the tensor.
            dtype: If supplied, the type of the values.

        Returns:
            The initialized tensor.

        """
        if dtype is None:
            dtype = self.dtype

        return np.array(
            self.rng.normal(
                loc,
                scale,
                ng.make_axes(axes).lengths),
            dtype=dtype)
