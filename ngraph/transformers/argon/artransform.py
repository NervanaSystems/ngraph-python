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
# import numpy as np

from neon import NervanaObject

from ngraph.transformers.base import Transformer
from ngraph.transformers.argon.mpihandle import MPIHandle


class ArgonTransformer(Transformer):
    """TODO."""

    # transformer_name = "argon"

    def __init__(self, **kwargs):
        super(ArgonTransformer, self).__init__(**kwargs)
        self.be = NervanaObject.be

    # allocators
    def empty(self, tensor_description):
        """
        TODO.

        Arguments:
          tensor_description: TODO

        Returns:

        """
        # TODO: Just to stop ar_backend error, should be replaced appropriately
        # Just a placeholder for now
        # bufshape = self.reshape(tensor_description.sizes)
        tensor_shape = tensor_description.sizes
        if len(tensor_shape) == 0:
            tensor_shape = 1
        elif len(tensor_shape) == 1:
            tensor_shape = (tensor_shape[0], 1)

        return self.be.empty(
            tensor_shape,
            dtype=tensor_description.dtype,
            persist_values=True)

    def nparray(self, tensor_description, array):
        """
        TODO.

        Arguments:
          tensor_description: TODO
          array: TODO
        """
        raise NotImplementedError()

    def rng(self, seed=None):
        """
        TODO.

        Arguments:
          seed: TODO
        """
        raise NotImplementedError()

    def tensor_view(self, tensor_description):
        """
        TODO.

        Arguments:
          tensor_description: TODO
        """
        raise NotImplementedError()

    def rng_uniform_tensor(self, rng, tensor_description, low, high):
        """
        TODO.

        Arguments:
          rng: TODO
          tensor_description: TODO
          low: TODO
          high: TODO
        """
        raise NotImplementedError()

    # Side-effects
    def fill(self, out, value):
        """
        TODO.

        Arguments:
          out: TODO
          value: TODO
        """
        raise NotImplementedError()

    def set_item(self, tensor, item, value):
        """
        TODO.

        Arguments:
          tensor: TODO
          item: TODO
          value: TODO
        """
        raise NotImplementedError()

    def rng_uniform(self, rng, low, high, out):
        """
        TODO.

        Arguments:
          rng: TODO
          low: TODO
          high: TODO
          out: TODO
        """
        raise NotImplementedError()

    def absolute(self, x, out):
        """
        TODO.

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    def add(self, x, y, out):
        """
        TODO.

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    # this function has
    def argmax(self, x, out, axis=0):
        """
        TODO.

        Arguments:
          x: TODO
          out: TODO
          axis: TODO
        """
        raise NotImplementedError()

    def argmin(self, x, axis, out):
        """
        TODO.

        Arguments:
          x: TODO
          axis: TODO
          out: TODO
        """
        raise NotImplementedError()

    def cos(self, x, out):
        """
        TODO.

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    def divide(self, x, y, out):
        """
        TODO.

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    def dot(self, x, y, out):
        """
        TODO.

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    def equal(self, x, y, out):
        """
        TODO.

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    def exp(self, x, out):
        """
        TODO.

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    def greater(self, x, y, out):
        """
        TODO.

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    def greater_equal(self, x, y, out):
        """
        TODO.

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    def less(self, x, y, out):
        """
        TODO.

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    def less_equal(self, x, y, out):
        """
        TODO.

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    def log(self, x, out):
        """
        TODO.

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    def max(self, x, axis, out):
        """
        TODO.

        Arguments:
          x: TODO
          axis: TODO
          out: TODO
        """
        raise NotImplementedError()

    def maximum(self, x, y, out):
        """
        TODO.

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    def min(self, x, axis, out):
        """
        TODO.

        Arguments:
          x: TODO
          axis: TODO
          out: TODO
        """
        raise NotImplementedError()

    def minimum(self, x, y, out):
        """
        TODO.

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    def multiply(self, x, y, out):
        """
        TODO.

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    def negative(self, x, out):
        """
        TODO.

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    def not_equal(self, x, y, out):
        """
        TODO.

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    def reciprocal(self, x, out):
        """
        TODO.

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    def sign(self, x, out):
        """
        TODO.

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    def sin(self, x, out):
        """
        TODO.

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    def sqrt(self, x, out):
        """
        TODO.

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    def square(self, x, out):
        """
        TODO.

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    def subtract(self, x, y, out):
        """
        TODO.

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    def sum(self, x, axis, out):
        """
        TODO.

        Arguments:
          x: TODO
          axis: TODO
          out: TODO
        """
        raise NotImplementedError()

    def tanh(self, x, out):
        """
        TODO.

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    def check_argon_error(self, err):
        """
        TODO.

        Arguments:
          err: TODO
        """
        raise NotImplementedError()

    def allreduce(self, x, out):
        """
        TODO.

        Arguments:
          x: TODO
          out: TODO
        """
        x_val = x.get()  # read data from Argon to CPU -- expensive!
        recv_buffer = MPIHandle().allreduceAvg(x_val)
        out.set(recv_buffer)
