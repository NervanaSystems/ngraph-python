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
from builtins import range

from ngraph.transformers.gpu.kernel import GPUKernel, pointer_from_td
from ngraph.transformers.gpu.float_ew2 import TensorDescriptionWrapper
from ngraph.transformers.gpu.kernels.cuda.copy_transpose import _get_copy_transpose_kernel
from ngraph.op_graph.axes import TensorDescription
from queue import Empty
import numpy as np
import pycuda.driver as drv

SLEEP_S = 0.1


def set_ipc_handle(op, shared_queue, handle):
    lock = drv.mem_alloc(1)
    drv.memset_d8(lock, 0, 1)
    buf_ipc_hdl = drv.mem_get_ipc_handle(handle)
    lock_ipc_hdl = drv.mem_get_ipc_handle(lock)
    shared_queue.put((buf_ipc_hdl, lock_ipc_hdl))
    return (lock)


def open_ipc_handle(shared_queue):
    print("inside open_ipc_handle(): shared_queue=")
    print(shared_queue)
    while True:
        try:
            (buf_ipc_hdl, lock_ipc_hdl) = shared_queue.get(timeout=SLEEP_S)
            buf_hdl = drv.IPCMemoryHandle(buf_ipc_hdl)
            lock = drv.IPCMemoryHandle(lock_ipc_hdl)
            return (buf_hdl, lock)
        except Exception as e:
            if isinstance(e, Empty):
                pass
            else:
                raise


def get_dimshuffle(dtype, shape, axes, src, dst):
    """
    Gets dimshuffle kernel and parameters for two same-sized tensors

    Arguments:
        dtype: tensor data type
        shape (tuple): source shape
        axes (tuple): new order of axes
        src (TensorDescriptionWrapper): source tensor
        dst (TensorDescriptionWrapper): dest tensor
    """
    kernel = _get_copy_transpose_kernel(dtype, shape, axes)
    params = [dst.td, src.td] + list(kernel.args)
    params = params + list(src.strides) + list(dst.strides)

    return (kernel, params)


class DimShuffleKernel(GPUKernel):
    """
    Kernel used to copy a tensor into another tensor with the same axes, but
    different order of dimensions. A transpose that supports any number or
    ordering of dimensions.

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU
        op (DimShuffle): Graph op being transformed into this kernel

    Attributes:
        kernel (pycuda.driver.Function): Compiled GPU kernel to execute this
            dimshuffle operation
        params (list): List of parameters to pass to kernel
    """

    def __init__(self, transformer, op):
        super(DimShuffleKernel, self).__init__(transformer)

        out = TensorDescriptionWrapper(op.tensor_description())
        (arg, ) = (_ for _ in op.call_info())
        in_tensor = TensorDescriptionWrapper(arg, ignore_layout=True)

        # Reshape the tensors in place with dimshuffle views
        in_tensor.shape = tuple(op.in_view.shape)
        in_tensor.strides = tuple(op.in_view.strides)
        out.shape = tuple(op.out_view.shape)
        out.strides = tuple(op.out_view.strides)

        dtype = out.dtype
        shape = in_tensor.shape
        axes = op.axis_order

        self.kernel, self.params = get_dimshuffle(dtype, shape, axes, in_tensor, out)

    def bind_buffers(self):
        """
        Binds GPU addresses of buffers to the kernel parameters. When kernels
        and initial parameters are generated, tensors have not yet been
        allocated so a placeholder is used for the memory addresses. This must
        be called before the first kernel run to bind the tensor addresses in
        GPU memory to the kernel parameters.
        """
        for index in range(len(self.params)):
            if isinstance(self.params[index], TensorDescription):
                self.params[index] = pointer_from_td(self.params[index])

        super(DimShuffleKernel, self).bind_buffers()

    def execute(self):
        """
        Calls the compiled DimShuffle kernel on the default stream.
        """
        self.kernel.prepared_async_call(self.kernel.grid, self.kernel.block,
                                        None, *self.params)

    def bind_flex_scales(self):
        pass


class FillKernel(GPUKernel):
    """
    Kernel used to fill a tensor with a scalar value.

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU
        tensor (TensorDescription): Tensor to fill
        value : Scalar value used to fill tensor

    Attributes:
        value : Scalar value to fill tensor
        out (GPUTensor): Tensor to fill with value
    """

    def __init__(self, transformer, tensor, value):
        super(FillKernel, self).__init__(transformer)

        self.value = value
        self.tensor = tensor

    def bind_buffers(self):
        """
        Get allocated GPU tensor for output
        """
        self.tensor = self.tensor.value.tensor
        super(FillKernel, self).bind_buffers()

    def execute(self):
        """
        Use memset driver functions to fill tensor with scalar
        Temporarily uses neon GPUTensor's fill method
        """
        self.tensor.fill(self.value)


class QueueSendKernel(GPUKernel):

    def __init__(self, transformer, send_op):
        super(QueueSendKernel, self).__init__(transformer)
        self.q = send_op.queue
        self.tensor = send_op.args[0].tensor_description()

    def bind_buffers(self):
        if isinstance(self.tensor, TensorDescription):
            self.tensor = self.tensor.value
        super(QueueSendKernel, self).bind_buffers()

    def execute(self):
        value = self.tensor.get(None)
        self.q.put(value)


class QueueRecvKernel(GPUKernel):
    """
    Kernel used to receive a tensor. The tensor's value can be
    a scalar, another tensor, or a numpy array

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU
        op (RecvKernel): Graph op being transformed into this kernel

    Attributes:
        tensor (GPUTensor): Dest tensor
    """

    def __init__(self, transformer, op):
        super(QueueRecvKernel, self).__init__(transformer)
        self.recv_op = op
        self.tensor = op.tensor_description()

    def bind_buffers(self):
        """
        Get allocated GPU tensor for output and potentially source value
        """
        if isinstance(self.tensor, TensorDescription):
            self.tensor = self.tensor.value
        super(QueueRecvKernel, self).bind_buffers()

    def execute(self):
        """
        Receive tensor
        """
        q = self.recv_op.queue
        x = q.get()

        if self.tensor.shape == ():
            self.tensor.tensor.fill(x)
        else:
            self.tensor.__setitem__(None, x)


class CudaScatterSendKernel(GPUKernel):

    def __init__(self, transformer, op):
        print("In class CudaScatterSendKernel init():transformer=")
        print(transformer)
        print("In class CudaScatterSendKernel init():op=")
        print(op)
        print("In class CudaScatterSendKernel init():op._shared_queues")
        print(op._shared_queues)
        super(CudaScatterSendKernel, self).__init__(transformer)
        self.op = op
        self.tensor = op.args[0].tensor_description()

    def bind_buffers(self):
        if isinstance(self.tensor, TensorDescription):
            self.tensor = self.tensor.value
        super(CudaScatterSendKernel, self).bind_buffers()
        self.send_ready = list()
        for i in range(len(self.op.to_id)):
            print("In class CudaScatterSendKernel bind_buffers():self.op")
            print(self.op)
            self.send_ready.append(
                set_ipc_handle(
                    self.op,
                    self.op._shared_queues[i],
                    self.tensor.tensor.gpudata))

    def execute(self):
        for i in range(len(self.op.to_id)):
            drv.memset_d8(self.send_ready[i], 1, 1)


class CudaScatterRecvKernel(GPUKernel):

    def __init__(self, transformer, op):
        print("In class CudaScatterRecvKernel init():transformer=")
        print(transformer)
        print("In class CudaScatterRecvKernel init():op")
        print(op)
        print("In class CudaScatterRecvKernel init():op_shared_queues=")
        print(op._shared_queues)
        super(CudaScatterRecvKernel, self).__init__(transformer)
        self.op = op
        self.tensor = op.tensor_description()
        (self.tnsr_ipc_hdl, self.sender_ready) = open_ipc_handle(op._shared_queues[op.idx])

    def bind_buffers(self):
        if isinstance(self.tensor, TensorDescription):
            self.tensor = self.tensor.value
        super(CudaScatterRecvKernel, self).bind_buffers()
        chunk_size = self.tensor.tensor.size * self.op.dtype.itemsize
        self.sender_buf = int(self.tnsr_ipc_hdl) + self.op.idx * chunk_size

    def execute(self):
        sender_ready = drv.from_device(self.sender_ready, (1,), np.int8)
        while (sender_ready == 0):
            sender_ready = drv.from_device(self.sender_ready, (1,), np.int8)
        drv.memcpy_dtod(
            self.tensor.tensor.gpudata,
            self.sender_buf,
            self.tensor.tensor.size * self.op.dtype.itemsize)
        drv.memset_d8(self.sender_ready, 0, 1)


class CudaGatherSendKernel(GPUKernel):

    def __init__(self, transformer, op):
        print("In class CudaGatherSendKernel init():transformer=")
        print(transformer)
        print("In class CudaGatherSendKernel init():op=")
        print(op)
        print("In class CudaGatherSendKernel init():shared_queues=")
        print(op._shared_queues)
        super(CudaGatherSendKernel, self).__init__(transformer)
        self.op = op
        self.tensor = op.args[0].tensor_description()
        (self.tnsr_ipc_hdl, self.send_ready) = open_ipc_handle(op._shared_queues[op.idx])

    def bind_buffers(self):
        if isinstance(self.tensor, TensorDescription):
            self.tensor = self.tensor.value
        super(CudaGatherSendKernel, self).bind_buffers()
        chunk_size = self.tensor.tensor.size * self.op.dtype.itemsize
        self.recvr_buf = int(self.tnsr_ipc_hdl) + self.op.idx * chunk_size

    def execute(self):
        # Push our fragment into its section of the larger recvr buffer, which assumes gather axis
        # is least contiguous.
        drv.memcpy_dtod(
            self.recvr_buf,
            self.tensor.tensor.gpudata,
            self.tensor.tensor.size * self.op.dtype.itemsize)
        drv.memset_d8(self.send_ready, 1, 1)


class CudaGatherRecvKernel(GPUKernel):

    def __init__(self, transformer, op):
        print("In class CudaGatherRecvKernel init():transformer=")
        print(transformer)
        print("In class CudaGatherRecvKernel init():op=")
        print(op)
        print("In class CudaGatherRecvKernel init():op._shared_queues=")
        print(op._shared_queues)
        super(CudaGatherRecvKernel, self).__init__(transformer)
        self.op = op
        self.tensor = op.tensor_description()

    def bind_buffers(self):
        if isinstance(self.tensor, TensorDescription):
            self.tensor = self.tensor.value
        super(CudaGatherRecvKernel, self).bind_buffers()
        self.sender_ready = list()
        for i in range(len(self.op.from_id)):
            self.sender_ready.append(
                set_ipc_handle(
                    self.op,
                    self.op._shared_queues[i],
                    self.tensor.tensor.gpudata))

    def execute(self):
        for i in range(len(self.op.from_id)):
            sender_ready = drv.from_device(self.sender_ready[i], (1,), np.int8)
            while (sender_ready == 0):
                sender_ready = drv.from_device(self.sender_ready[i], (1,), np.int8)
            drv.memset_d8(self.sender_ready[i], 0, 1)


class RngFillKernel(GPUKernel):
    """
    Kernel used to fill a tensor with a random distribution value.

    Arguments:
        transformer (GPUTransformer): GPU transformer with kernel generator and runtime driver
        td (TensorDescription): Tensor to fill
        distribution (str): type of random distribution to use,
                            can be either 'uniform' or 'normal'
        params (dict): distribution specific parameters

    Attributes:
        value : Scalar value to fill tensor
        out (GPUTensor): Tensor to fill with value
    """

    def __init__(self, transformer, td, distribution, params):
        super(RngFillKernel, self).__init__(transformer)

        self.distribution = distribution
        self.params = params
        self.out = td

    def bind_buffers(self):
        """
        Get allocated GPU tensor for output
        """
        self.out = self.out.value.tensor
        super(RngFillKernel, self).bind_buffers()

    def execute(self):
        """
        Use memset driver functions to fill tensor with scalar
        """
        if self.distribution == 'uniform':
            self.transformer.runtime.pcg.fill_uniform(self.out)
            self.out[:] = (self.out * (self.params['high'] - self.params['low']) +
                           self.params['low'])
        elif self.distribution == 'normal':
            self.transformer.runtime.pcg.fill_normal(self.out)
            self.out[:] = self.out * self.params['scale'] + self.params['loc']


class SetItemKernel(GPUKernel):
    """
    Kernel used set all or part of a tensor with a value. Value can be
    a scalar, another tensor, or a numpy array

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU
        op (SetItemOneDOp): Graph op being transformed into this kernel

    Attributes:
        tensor (GPUTensor): Dest tensor
        value: Source scalar, numpy array, or tensor
        item (slice): Slice to apply to dest tensor
    """

    def __init__(self, transformer, op):
        super(SetItemKernel, self).__init__(transformer)

        self.tensor, self.value = (_ for _ in op.call_info())
        self.item = op.item

        # Use copy transpose kernel for unsupported cases
        if len(self.tensor.strides) > 0 and np.min(self.tensor.strides) < 0 and self.item is None:
            dtype = self.tensor.dtype
            shape = self.tensor.shape
            axes = range(len(self.tensor.shape))

            self.kernel, self.params = get_dimshuffle(dtype, shape, tuple(axes),
                                                      TensorDescriptionWrapper(self.value),
                                                      TensorDescriptionWrapper(self.tensor))
        else:
            self.kernel = None

    def bind_buffers(self):
        """
        Get allocated GPU tensor for output and potentially source value
        """
        if isinstance(self.tensor, TensorDescription):
            self.tensor = self.tensor.value
        if isinstance(self.value, TensorDescription):
            self.value = self.value.value.tensor

        if self.kernel is not None:
            for index in range(len(self.params)):
                if isinstance(self.params[index], TensorDescription):
                    self.params[index] = pointer_from_td(self.params[index])

        super(SetItemKernel, self).bind_buffers()

    def execute(self):
        """
        Run kernel to copy into tensor
        Temporarily using the neon GPUTensor implementation
        """
        if self.kernel is not None:
            self.kernel.prepared_async_call(self.kernel.grid, self.kernel.block,
                                            None, *self.params)
        elif self.tensor.shape == ():
            self.tensor.tensor.fill(self.value)
        else:
            self.tensor.__setitem__(self.item, self.value)


class FlexFillKernel(FillKernel):
    """
    Flex version of FillKernel
    """

    def __init__(self, transformer, tensor, value):
        super(FlexFillKernel, self).__init__(transformer, tensor, value)

        self.flex_entry = self.tensor.value.flex_entry
        self.output_flex_ids = [self.flex_entry.flex_id]

    def execute(self):
        val = int(self.value / self.scale)  # flex value storage

        # if overflow, fill tensor with clipped value and set maxabs to clipped value
        if val > self.flex_entry.dtype.pclip:
            # overflow on positive side
            clipped = int(self.flex_entry.dtype.pclip)
            self.tensor.fill(clipped)  # tensor is int for flex storage
            self.maxabs = clipped  # positive, scalar value
        elif val < self.flex_entry.dtype.nclip:
            # overflow on negative side
            clipped = int(self.flex_entry.dtype.nclip)
            self.tensor.fill(clipped)
            self.maxabs = abs(clipped)
        else:
            # no overflow
            self.tensor.fill(val)
            self.maxabs = abs(val)

    def bind_flex_scales(self):
        self.scale = self.flex_entry.scale


class FlexRngFillKernel(RngFillKernel):
    """
    Flex version of RngFillKernel
    """

    def __init__(self, transformer, td, distribution, params):
        super(FlexRngFillKernel, self).__init__(transformer, td, distribution, params)

        # save flex entry for bind_flex_scales
        self.flex_entry = td.value.flex_entry
        # output flex ids for autoflex to manage
        self.output_flex_ids = [self.flex_entry.flex_id]

    def execute(self):
        # self.out.dtype is int16, which is not supported by fill_uniform
        # generate floating point random values, then apply flex scale
        out_float = self.out.astype(np.float32)
        if self.distribution == 'uniform':
            self.transformer.runtime.pcg.fill_uniform(out_float)
            self.out[:] = ((out_float * (self.params['high'] - self.params['low']) +
                            self.params['low']) / self.scale).astype(self.out.dtype)
        elif self.distribution == 'normal':
            self.transformer.runtime.pcg.fill_normal(out_float)
            self.out[:] = ((out_float * self.params['scale'] + self.params['loc'])
                           / self.scale).astype(self.out.dtype)

    def bind_flex_scales(self):
        self.scale = self.flex_entry.scale
