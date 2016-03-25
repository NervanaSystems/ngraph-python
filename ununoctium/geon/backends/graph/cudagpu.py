from contextlib import contextmanager

import pycuda.driver as cuda
import numpy as np
from geon.backends.graph.storage import Store


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
        yield(context)
    finally:
        if context is not None:
            context.pop()
        clear_context_caches()

@contextmanager
def cuda_context(context=None):
    """
    Enable a particular CUDA context.
    :param context:
    :return:
    """
    try:
        if context is not None:
            context.push()
        yield(context)
    finally:
        if context is not None:
            cuda.Context.pop()


#TODO: Switch to the new Storage API
class GPUStorage(Store):
    def __init__(self, context, dtype, size, alloc=True):
        self.allocation = None
        self.context = context
        self.__dtype = np.dtype(dtype)
        self.__itemsize = self.__dtype.itemsize
        self.__size = size
        self.gpudata = None
        self.pitch = None
        if alloc:
            self.alloc()


    def alloc(self):
        if self.allocation is None:
            with cuda_context(self.context):
                self.gpudata = cuda.mem_alloc(self.__itemsize*self.__size)


    def free(self):
        if self.allocation is not None:
            self.allocation.free()
            self.allocation = None


    def __del__(self):
        self.free()


    @property
    def dtype(self):
        return self.__dtype

    @property
    def size(self):
        return self.__size


    def set(self, ary, offset=0):
        with cuda_context(self.context):
            cuda.memcpy_htod_async(int(self.gpudata)+offset*self.__itemsize, ary, None)

    def get(self, ary=None, offset=0):
        if ary is None:
            ary = np.empty((self.size-offset,), self.dtype)
        else:
            assert self.dtype is ary.dtype
        with cuda_context(self.context):
            cuda.memcpy_dtoh_async(ary, int(self.gpudata)+offset*self.__itemsize, None)
        return ary


class GraphTensor(object):
    def __init__(self, storage, offset, dtype, shape, strides):
        self.storage = storage
        self.offset = offset
        self.dtype = dtype
        self.shape = shape
        self.strides = strides

    def element_offset(self, index):
        offset = self.offset
        for i,stride in zip(index, self.strides):
            offset += i*stride
        return offset


    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        if isinstance(index, tuple):
            ary = np.array([0], dtype=self.dtype)
            storage.get(ary=ary,offset=self.element_offset(index))
            return ary[0]

    def __setitem__(self, index, value):
        if not isinstance(index, tuple):
            index = (index,)
        storage.set(np.array([value], dtype=self.dtype), self.element_offset(index))
        return value

