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
from builtins import range

from geon.op_graph.op_graph import AllocationOp
from geon.transformers.base import Transformer


class NumPyTransformer(Transformer):

    def __init__(self, **kargs):
        super(NumPyTransformer, self).__init__(**kargs)

    # allocators
    def make_raw_buffer(self, size):
        return bytearray(size)
    
    def fill_tensor_in(self, tensor_description, tensor):
        view = self.tensor_view(tensor_description)
        if view.shape:
            view[:] = tensor
        else:
            view = tensor
        return view
        
    def tensor_view(self, tensor_description):
        return np.ndarray(
                shape=tensor_description.shape,
                dtype=tensor_description.dtype,
                buffer=tensor_description.buffer.data,
                offset=tensor_description.offset,
                strides=tensor_description.strides)

    def nparray(self, tensor_description, array):
        tensor = self.tensor_view(tensor_description)
        tensor[:] = array
        return tensor

    def rng(self, seed=None):
        return np.random.RandomState(seed=seed)


    def rng_normal_tensor(self, rng, tensor_description, loc, scale):
        tensor = rng.normal(
                    loc, scale, tensor_description.sizes).astype(
                    tensor_description.dtype)
        return self.fill_tensor_in(tensor_description, tensor)

    def rng_uniform_tensor(self, rng, tensor_description, low, high):
        tensor =  rng.uniform(
            low, high, tensor_description.sizes).astype(
            tensor_description.dtype)
        return self.fill_tensor_in(tensor_description, tensor)

    # Side-effects
    def fill(self, out, value):
        out.fill(value)

    def set_item(self, tensor, item, value):
        tensor.__setitem__(item, value)

    # Operations
    def absolute(self, x, out):
        np.abs(x, out=out)

    def add(self, x, y, out):
        np.add(x, y, out=out)

    def argmax(self, x, out):
        np.ndarray.argmax(x, 0, out)

    def argmin(self, x, out):
        np.ndarray.argmin(x, 0, out)

    def cos(self, x, out):
        np.cos(x, out=out)

    def divide(self, x, y, out):
        np.divide(x, y, out=out)

    def dot(self, x, y, out):
        if not out.flags.c_contiguous:
            t = x
            x = y.T
            y = t.T
            out = out.T
        np.dot(x, y, out)

    def equal(self, x, y, out):
        return np.equal(x, y, out=out)

    def exp(self, x, out):
        np.exp(x, out=out)

    def greater(self, x, y, out):
        np.greater(x, y, out=out)

    def greater_equal(self, x, y, out):
        np.greater_equal(x, y, out=out)

    def less(self, x, y, out):
        np.less(x, y, out=out)

    def less_equal(self, x, y, out):
        np.less_equal(x, y, out=out)

    def log(self, x, out):
        np.log(x, out=out)

    def max(self, x, axis, out):
        #print '=========== MAX ==============='
        #print id(x.data), id(out.data) 
        #print id(out.data)
        #print id(x.data)
        np.max(x, axis, out=out)
        #print '=========== AFTER ==============='
        #print out
        #print id(out.data)
        #print '=========== DONE MAX ==============='

    def maximum(self, x, y, out):
        np.maximum(x, y, out=out)

    def min(self, x, axis, out):
        np.min(x, axis, out=out)

    def minimum(self, x, y, out):
        np.minimum(x, y, out=out)

    def multiply(self, x, y, out):
        np.multiply(x, y, out=out)

    def negative(self, x, out):
        np.negative(x, out=out)

    def not_equal(self, x, y, out):
        np.not_equal(x, y, out=out)

    def onehot(self, x, out):
        out[:] = 0
        for i in range(len(x)):
            out[x[i], i] = 1

    def reciprocal(self, x, out):
        np.reciprocal(x, out=out)

    def sign(self, x, out):
        np.sign(x, out=out)

    def sin(self, x, out):
        np.sin(x, out=out)

    def sqrt(self, x, out):
        np.sqrt(x, out=out)

    def square(self, x, out):
        np.square(x, out=out)

    def subtract(self, x, y, out):
        np.subtract(x, y, out=out)

    def sum(self, x, axis, out):
        np.sum(x, axis=axis, out=out)

    def tanh(self, x, out):
        np.tanh(x, out=out)


class NPNormal(AllocationOp):

    def __init__(self, rng, loc, scale, **kargs):
        super(NPNormal, self).__init__(args=(rng,), **kargs)
        self.loc = loc
        self.scale = scale

    def compute_tensor_axes_info(self):
        rng, = self.args
        tensor_axes_info = super(NPNormal, self).compute_tensor_axes_info()
        tensor_axes_info.alloc = lambda evaluator, tensor_description: evaluator.rng_normal_tensor(
            rng, tensor_description, self.loc, self.scale)


class NPUniform(AllocationOp):

    def __init__(self, rng, low, high, **kargs):
        super(NPUniform, self).__init__(args=(rng,), **kargs)
        self.low = low
        self.high = high

    def compute_tensor_axes_info(self):
        rng, = self.args
        tensor_axes_info = super(NPUniform, self).compute_tensor_axes_info()
        tensor_axes_info.alloc = lambda evaluator, tensor_description: \
            evaluator.rng_uniform_tensor(rng, tensor_description, self.low, self.high)
