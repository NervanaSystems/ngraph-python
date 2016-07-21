import numpy as np

from geon.backends.graph.transform import Transformer, AllocationOp, Visitor

import argon.neon_backend.ar_backend
from argon.neon_backend.ar_backend import ArBackend

from neon import NervanaObject
from neon.backends.backend import OpTreeNode 

from mpi4py import MPI

comm = MPI.COMM_WORLD

WITHNP = False

from mpi4py import MPI

comm = MPI.COMM_WORLD

class ArgonTransformer(Transformer):
        
    def __init__(self, **kargs):
        super(ArgonTransformer, self).__init__(**kargs)
        self.be = NervanaObject.be 
        
        
    # allocators
    def empty(self, tensor_description):
        # TODO: Just to stop ar_backend error, should be replaced appropriately
        ## Just a placeholder for now
        # bufshape = self.reshape(tensor_description.sizes)
        tensor_shape = tensor_description.sizes
        if len(tensor_shape) == 0:
            tensor_shape = 1
        elif len(tensor_shape) == 1:
            tensor_shape = (tensor_shape[0],1)

        return self.be.empty(tensor_shape, dtype=tensor_description.dtype, persist_values=True)

    def nparray(self, tensor_description, array):
        raise NotImplementedError()

    def rng(self, seed=None):
        raise NotImplementedError()

    def tensor_view(self, tensor_description):
        raise NotImplementedError()

    def rng_uniform_tensor(self, rng, tensor_description, low, high):
        raise NotImplementedError()

    # Side-effects
    def fill(self, out, value):
        raise NotImplementedError()

    def set_item(self, tensor, item, value):
        raise NotImplementedError()

    def rng_uniform(self, rng, low, high, out):
        raise NotImplementedError()

    def absolute(self, x, out):
        raise NotImplementedError()

    def add(self, x, y, out):
        raise NotImplementedError()
                
    # this function has 
    def argmax(self, x, out, axis=0):
        raise NotImplementedError()

    def argmin(self, x, axis, out):
        raise NotImplementedError()

    def cos(self, x, out):
        raise NotImplementedError()

    def divide(self, x, y, out):
        raise NotImplementedError()

    def dot(self, x, y, out):
        raise NotImplementedError()
        
    def equal(self, x, y, out):
        raise NotImplementedError()

    def exp(self, x, out):
        raise NotImplementedError()

    def greater(self, x, y, out):
        raise NotImplementedError()

    def greater_equal(self, x, y, out):
        raise NotImplementedError()

    def less(self, x, y, out):
        raise NotImplementedError()

    def less_equal(self, x, y, out):
        raise NotImplementedError()
        
    def log(self, x, out):
        raise NotImplementedError()

    def max(self, x, axis, out):
        raise NotImplementedError()

    def maximum(self, x, y, out):
        raise NotImplementedError()

    def min(self, x, axis, out):
        raise NotImplementedError()

    def minimum(self, x, y, out):
        raise NotImplementedError()

    def multiply(self, x, y, out):
        raise NotImplementedError()

    def negative(self, x, out):
        raise NotImplementedError()

    def not_equal(self, x, y, out):
        raise NotImplementedError()

    def reciprocal(self, x, out):
        raise NotImplementedError()

    def sign(self, x, out):
        raise NotImplementedError()

    def sin(self, x, out):
        raise NotImplementedError()

    def sqrt(self, x, out):
        raise NotImplementedError()

    def square(self, x, out):
        raise NotImplementedError()

    def subtract(self, x, y, out):
        raise NotImplementedError()

    def sum(self, x, axis, out):
        raise NotImplementedError()

    def tanh(self, x, out):
        raise NotImplementedError()

    def check_argon_error(self, err):
        err_code = AMStatusCode(err.code)
        if err_code != AMStatusCode.S_OK:
            raise RuntimeError("Argon error: {}".format(err_code))

    def allreduce(self, x, out):
        x_val = x.get()  # read data from Argon to CPU -- expensive!
        recv_buffer = np.zeros(shape=x.shape, dtype=x.dtype)
        comm.Allreduce(x_val, recv_buffer, op=MPI.SUM)
        recv_buffer = recv_buffer / comm.Get_size()  # Normalize the results to the number of MPI threads
        out[:] = recv_buffer


class ArgonUniform(AllocationOp):
    def __init__(self, rng, low, high, **kargs):
        super(ArgonUniform, self).__init__(args=(rng,), **kargs)
        self.low = low
        self.high = high

    def compute_tensor_axes_info(self):
        rng, = self.args
        tensor_axes_info = super(ArgonUniform, self).compute_tensor_axes_info()
        tensor_axes_info.alloc = lambda evaluator, tensor_description: evaluator.rng_uniform_tensor(rng, tensor_description, self.low, self.high)


class ArgonTransformVisitor(Visitor):
    def visit_uniform(self, uniform, low, high, rng):
        pass