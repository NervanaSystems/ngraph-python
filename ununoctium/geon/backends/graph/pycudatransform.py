import numpy as np
from contextlib import contextmanager

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath

from geon.backends.graph.transform import Op, Transformer

# Initialize CUDA
cuda.init()


@contextmanager
def cuda_device_context(device=None):
    """
    Get a CUDA context for a device.
    :param device:
    :return:
    """
    from pycuda.tools import make_default_context, clear_context_caches
    context = None
    try:
        if device is None:
            context = make_default_context()
        else:
            context = cuda.Device(device).make_context()
        yield (context)
    finally:
        if context is not None:
            context.pop()
        clear_context_caches()


# TODO This is mostly historical, and not up to date with the current Transformer
class PyCUDATransformer(Transformer):
    """
    Uses PuCUDA to evaluate.  Not fully tested; PyCUDA does not expose all the NumPy API.
    """

    def __init__(self, **kvargs):
        super(PyCUDATransformer, self).__init__(**kvargs)

    def evaluate(self, **kvargs):
        with cuda_device_context():
            return super(PyCUDATransformer, self).evaluate(**kvargs)

    # allocations
    def empty(self, tensor_description):
        return cumath.empty(tensor_description.sizes, tensor_description.dtype)

    def ones(self, tensor_description):
        result = self.empty(tensor_description)
        result.fill(1.0)
        return result

    def zeros(self, tensor_description):
        return gpuarray.zeros(tensor_description.sizes, tensor_description.dtype)

    # Operations

    def absolute(self, x, out):
        cumath.fabs(x, out=out)

    def add(self, x, y, out):
        x._axpbyz(1, y, 1, out)

    def cos(self, x, out):
        cumath.cos(x, out=out)

    def dot(self, x, y, out):
        cumath.dot(x, y, out=out)

    def exp(self, x, out):
        cumath.exp(x, out=out)

    def log(self, x, out):
        cumath.log(x, out=out)

    def maximum(self, x, y, out):
        cumath.maximum(x, y, out=out)

    def minimum(self, x, y, out):
        cumath.minimum(x, y, out=out)

    def multiply(self, x, y, out):
        if isinstance(x, gpuarray.GPUArray):
            if isinstance(y, gpuarray.GPUArray):
                x._elwise_multiply(y, out=out)
                return
            x._axpbz(y, 0, out)
            return
        elif isinstance(y, gpuarray.GPUArray):
            y._axpbz(x, 0, out)
            return
        else:
            out[:] = x * y

    def negative(self, x, out):
        x._axpbz(-1, 0.0, out)

    def reciprocal(self, x, out):
        x._rdiv_scalar(1.0, out)

    def sig(self, x, out):
        self.negative(x, out=out)
        cumath.exp(out, out=out)
        # Add one
        out._axpbz(1.0, 1.0, out=out)
        out._rdiv_scalar(1.0, out=out)

    def sign(self, x, out):
        out.set(np.sign(x.get()))

    def sin(self, x, out):
        cumath.sin(x, out=out)

    def sqrt(self, x, out):
        cumath.sqrt(x, out=out)

    def square(self, x, out):
        self.multiply(x, x, out)

    def subtract(self, x, y, out):
        x._axpbyz(1, y, 1, out)

    def tanh(self, x, out):
        cumath.tanh(x, out=out)
