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

from ngraph.transformers.gpu.kernel import GPUKernel
from ngraph.transformers.gpu.float_ew2 import TensorDescriptionWrapper
from ngraph.transformers.gpu.kernels.cuda.copy_transpose import _get_copy_transpose_kernel
from ngraph.op_graph.axes import TensorDescription
from queue import Empty
import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.driver import event_flags


SLEEP_S = 0.1
ITEMS_PER_THREAD = 32


def set_ipc_handle(op, shared_queue, handle, local=False):
    lock = drv.mem_alloc(1)
    drv.memset_d8(lock, 0, 1)
    if local:
        buf_ipc_hdl = int(handle)
        lock_ipc_hdl = int(lock)
    else:
        buf_ipc_hdl = drv.mem_get_ipc_handle(handle)
        lock_ipc_hdl = drv.mem_get_ipc_handle(lock)
    shared_queue.put((local, buf_ipc_hdl, lock_ipc_hdl))
    return (lock)


def open_ipc_handle(shared_queue):
    while True:
        try:
            (is_local, buf_ipc_hdl, lock_ipc_hdl) = shared_queue.get(timeout=SLEEP_S)
            if is_local:
                return buf_ipc_hdl, lock_ipc_hdl
            else:
                buf_hdl = drv.IPCMemoryHandle(buf_ipc_hdl)
                lock = drv.IPCMemoryHandle(lock_ipc_hdl)
                return (buf_hdl, lock)
        except Exception as e:
            if isinstance(e, Empty):
                pass
            else:
                raise


def _reduction_kernel(op):
    kernel_code_template = """
        #define ITEMS_PER_THREAD 32

        __global__ void float_accum(float *dest, float *scratch, int max_size,
                                    int num_scratch_arrays, int scratch_array_size)
        {{
            float4 dest_regs;
            float4 scratch_regs;
            int offset = (blockIdx.x * ITEMS_PER_THREAD * blockDim.x) + (threadIdx.x * 4);
            float ndevs = num_scratch_arrays + 1;

            #pragma unroll
            for(int i = 0; i < ITEMS_PER_THREAD; i+=4)
            {{
                if(offset < max_size)
                {{
                    dest_regs = *((float4*)(&(dest[offset])));
                }}

                for(int array_id = 0; array_id < num_scratch_arrays; array_id++)
                {{
                    int scratch_offset = (array_id * scratch_array_size) + offset;

                    if(offset < max_size)
                    {{
                        scratch_regs = *((float4*)(&(scratch[scratch_offset])));

                        dest_regs.x += scratch_regs.x;
                        dest_regs.y += scratch_regs.y;
                        dest_regs.z += scratch_regs.z;
                        dest_regs.w += scratch_regs.w;
                    }}
                }}

                {mean_code}

                if(offset < max_size)
                {{
                    *((float4*)(&(dest[offset]))) = dest_regs;
                }}

                offset += (blockDim.x * 4);
            }}
        }}
        """
    if op == "mean":
        mean_code = """
                    if(offset < max_size)
                    {{
                        dest_regs.x /= ndevs;
                        dest_regs.y /= ndevs;
                        dest_regs.z /= ndevs;
                        dest_regs.w /= ndevs;
                    }}
                    """
    else:
        mean_code = "(void)ndevs;"

    kernel_code = kernel_code_template.format(mean_code=mean_code)
    _float_accum_kernel = SourceModule(kernel_code)

    kernel = _float_accum_kernel.get_function("float_accum")
    kernel.prepare("PPiii")
    return kernel


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

        out = TensorDescriptionWrapper(self.transformer, op.tensor_description())
        (arg, ) = (_ for _ in op.call_info())
        in_tensor = TensorDescriptionWrapper(self.transformer, arg, ignore_layout=True)

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
                self.params[index] = self.pointer_from_td(self.params[index])

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
        self.tensor = self.tensor_view_from_td(self.tensor).tensor
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
            self.tensor = self.tensor_view_from_td(self.tensor)
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
            self.tensor = self.tensor_view_from_td(self.tensor)
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
        super(CudaScatterSendKernel, self).__init__(transformer)
        self.op = op
        self.tensor = op.args[0].tensor_description()

    def bind_buffers(self):
        if isinstance(self.tensor, TensorDescription):
            self.tensor = self.tensor_view_from_td(self.tensor)
        # if input buffer changes we need to make sure to set new ipc handle and
        # signal the recv kernel to get the new ipc handle.
        # Assuming bind_buffers() is called once and the tensor does not change
        super(CudaScatterSendKernel, self).bind_buffers()
        self.send_ready = list()
        for i, to_id in enumerate(self.op.to_id):
            self.send_ready.append(
                set_ipc_handle(
                    self.op,
                    self.op._shared_queues[i],
                    self.tensor.tensor.gpudata,
                    local=self.op.metadata['device_id'] == int(to_id)))

    def execute(self):
        for i in range(len(self.op.to_id)):
            drv.memset_d8(self.send_ready[i], 1, 1)


class CudaScatterRecvKernel(GPUKernel):

    def __init__(self, transformer, op):
        super(CudaScatterRecvKernel, self).__init__(transformer)
        self.op = op
        self.tensor = op.tensor_description()

    def bind_buffers(self):
        if isinstance(self.tensor, TensorDescription):
            self.tensor = self.tensor_view_from_td(self.tensor)
        (self.tnsr_ipc_hdl, self.sender_ready) = open_ipc_handle(self.op._shared_queues
                                                                 [self.op.idx])
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
        super(CudaGatherSendKernel, self).__init__(transformer)
        self.op = op
        self.tensor = op.args[0].tensor_description()
        self.recvr_buf = None

    def bind_buffers(self):
        if isinstance(self.tensor, TensorDescription):
            self.tensor = self.tensor_view_from_td(self.tensor)
        super(CudaGatherSendKernel, self).bind_buffers()

    def execute(self):
        if self.recvr_buf is None:
            (self.tnsr_ipc_hdl, self.send_ready) = open_ipc_handle(self.op._shared_queues
                                                                   [self.op.idx])
            chunk_size = self.tensor.tensor.size * self.op.dtype.itemsize
            self.recvr_buf = int(self.tnsr_ipc_hdl) + self.op.idx * chunk_size

        # Push our fragment into its section of the larger recvr buffer, which assumes gather axis
        # is least contiguous.
        drv.memcpy_dtod(
            self.recvr_buf,
            self.tensor.tensor.gpudata,
            self.tensor.tensor.size * self.op.dtype.itemsize)
        drv.memset_d8(self.send_ready, 1, 1)


class CudaGatherRecvKernel(GPUKernel):

    def __init__(self, transformer, op):
        super(CudaGatherRecvKernel, self).__init__(transformer)
        self.op = op
        self.tensor = op.tensor_description()

    def bind_buffers(self):
        super(CudaGatherRecvKernel, self).bind_buffers()
        if isinstance(self.tensor, TensorDescription):
            self.tensor = self.tensor_view_from_td(self.tensor)
        self.sender_ready = list()
        for i, from_id in enumerate(self.op.from_id):
            self.sender_ready.append(
                set_ipc_handle(
                    self.op,
                    self.op._shared_queues[i],
                    self.tensor.tensor.gpudata,
                    local=int(from_id) == self.op.metadata['device_id']))

    def execute(self):
        for i in range(len(self.op.from_id)):
            sender_ready = drv.from_device(self.sender_ready[i], (1,), np.int8)
            while (sender_ready == 0):
                sender_ready = drv.from_device(self.sender_ready[i], (1,), np.int8)
            drv.memset_d8(self.sender_ready[i], 0, 1)


class CudaAllReduceKernel(GPUKernel):

    def __init__(self, transformer, op):
        super(CudaAllReduceKernel, self).__init__(transformer)
        self.op = op
        self.tensor = op.tensor_description()
        self.device_id = transformer.device_id
        self.event = drv.Event(flags=event_flags.INTERPROCESS | event_flags.DISABLE_TIMING)
        self.stream = drv.Stream()
        self.output_buff_dict = {}
        self.scratch_buff_dict = {}
        self.event_buff_dict = {}
        self.init_buffers()

    def init_buffers(self):
        shape = self.op.args[0].tensor_description().shape
        dtype = self.op.args[0].tensor_description().dtype

        # Allocate output and scratch buffers
        self.output_buff = gpuarray.zeros(shape, dtype)
        self.scratch_buff = gpuarray.zeros(shape, dtype)

        self.output_buff_dict[self.device_id] = self.output_buff.gpudata
        self.scratch_buff_dict[self.device_id] = self.scratch_buff.gpudata

        # Allocate IPC handles
        output_ipc_hdl = drv.mem_get_ipc_handle(self.output_buff.gpudata)
        scratch_ipc_hdl = drv.mem_get_ipc_handle(self.scratch_buff.gpudata)
        event_ipc_hdl = self.event.ipc_handle()

        # Put handles in queues
        for i in self.op.shared_queues.keys():
            if i != self.device_id:
                self.op.shared_queues[i].put(
                    (self.device_id, output_ipc_hdl, scratch_ipc_hdl, event_ipc_hdl))

        # Get handles from others
        q = self.op.shared_queues[self.device_id]
        for i in range(len(self.op.shared_queues) - 1):
            peer_id, output_ipc_hdl, scratch_ipc_hdl, event_ipc_hdl = q.get()
            output_hdl = drv.IPCMemoryHandle(output_ipc_hdl)
            scratch_hdl = drv.IPCMemoryHandle(scratch_ipc_hdl)
            event_hdl = drv.Event.from_ipc_handle(event_ipc_hdl)
            self.output_buff_dict[peer_id] = output_hdl
            self.scratch_buff_dict[peer_id] = scratch_hdl
            self.event_buff_dict[peer_id] = event_hdl

    def bind_buffers(self):
        if isinstance(self.tensor, TensorDescription):
            self.tensor = self.tensor.value
        super(CudaAllReduceKernel, self).bind_buffers()
        self.input_tensor = self.op.args[0].value

    def execute(self):
        ndevs = len(self.op.device_ids)
        size = self.input_tensor.tensor.size
        dtype = self.input_tensor.dtype
        segment_size = int(size / ndevs)
        if ((segment_size * ndevs) < size):
            segment_size += 1

        # Align segment size to 16 bytes
        if (segment_size & 0x03):
            segment_size = (segment_size & (~0x03)) + 4

        # Determine GPU active mask based on segment size
        num_active = int(size / segment_size)
        if ((segment_size * num_active) < size):
            num_active += 1

        # Copy tensor to output buffer
        drv.memcpy_dtod(
            self.output_buff.gpudata,
            self.input_tensor.tensor.gpudata,
            size * dtype.itemsize)

        # Send each GPU its assigned segment
        for peer_id in range(ndevs):
            if (peer_id == self.device_id):
                continue

            # Only send if peer is active
            if (peer_id >= num_active):
                continue

            # Compute size and offset of this peer's segment
            peer_segment_size = segment_size
            peer_segment_offset = peer_id * segment_size

            if (self.device_id > peer_id):
                peer_scratch_offset = segment_size * (self.device_id - 1)
            else:
                peer_scratch_offset = segment_size * self.device_id

            if ((peer_id + 1) == num_active):
                peer_segment_size = size - peer_segment_offset

            # Enqueue peer to peer memcpy
            src = int(self.output_buff_dict.get(self.device_id)) + \
                peer_segment_offset * dtype.itemsize
            scratch = int(self.scratch_buff_dict.get(peer_id)) + \
                peer_scratch_offset * dtype.itemsize

            drv.memcpy_dtod_async(scratch, src,
                                  peer_segment_size * dtype.itemsize,
                                  self.stream)

        # Record event in stream
        self.event.record(self.stream)

        # Sync with other devices
        self.process_sync()

        # Wait for other GPUs events
        for peer_id in self.op.device_ids:
            if (peer_id == self.device_id):
                continue
            self.stream.wait_for_event(self.event_buff_dict[peer_id])

        segment_offset = self.device_id * segment_size
        this_segment_size = segment_size
        if ((self.device_id + 1) == num_active):
            this_segment_size = size - segment_offset

        src = int(self.output_buff_dict.get(self.device_id)) + \
            segment_offset * dtype.itemsize

        # Sum received peer segments
        block_size = 1024
        grid_size = int(this_segment_size / (block_size * ITEMS_PER_THREAD))
        if ((grid_size * block_size * ITEMS_PER_THREAD) < this_segment_size):
            grid_size += 1

            # Perform reduction operation
            if (self.device_id < num_active):
                num_arrays = ndevs - 1
                params = [src, self.scratch_buff_dict[self.device_id],
                          this_segment_size, num_arrays, segment_size]
                grid_dim = (grid_size, 1, 1)
                block_dim = (block_size, 1, 1)
                kernel = _reduction_kernel(self.op.reduce_func)
                kernel.prepared_async_call(grid_dim, block_dim, self.stream, *params)

                # Send other GPUs this GPU's assigned segment
                for peer_id in self.op.device_ids:
                    if (peer_id == self.device_id):
                        continue

                    # Enqueue peer to peer memcpy
                    dst = int(self.output_buff_dict.get(peer_id)) + \
                        segment_offset * dtype.itemsize
                    drv.memcpy_dtod_async(dst, src,
                                          this_segment_size * dtype.itemsize,
                                          self.stream)

            self.event.record(self.stream)

            self.process_sync()

            # Wait for other GPUs events
            for peer_id in self.op.device_ids:
                if (peer_id == self.device_id):
                    continue
                self.event_buff_dict[peer_id].synchronize()
            self.event.synchronize()

            drv.memcpy_dtod_async(
                self.tensor.tensor.gpudata,
                self.output_buff.gpudata,
                size * dtype.itemsize,
                self.stream)

            # This sync is only needed if we call this kernel 'synchronously'
            # if the assumption is that another kernel is called right after,
            # and uses the same streams as us, then we can remove this and
            # rely on the next kernel being put into our stream.

            # Record event in stream
            self.event.record(self.stream)

            # Sync with other devices
            self.process_sync()

            # Wait for other GPUs events
            for peer_id in self.op.device_ids:
                if (peer_id == self.device_id):
                    continue
                self.event_buff_dict[peer_id].synchronize()
            self.event.synchronize()

    def process_sync(self):
        x = None
        queues = self.op.shared_queues
        ndevs = len(self.op.device_ids)
        for peer_id in self.op.device_ids:
            if peer_id != self.device_id:
                queues[peer_id].put(x)
        for i in range(ndevs - 1):
            x = queues[self.device_id].get()


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
        self.out = self.tensor_view_from_td(self.out).tensor
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

            self.kernel, self.params = \
                get_dimshuffle(dtype, shape, tuple(axes),
                               TensorDescriptionWrapper(self.transformer, self.value),
                               TensorDescriptionWrapper(self.transformer, self.tensor))
        else:
            self.kernel = None

    def bind_buffers(self):
        """
        Get allocated GPU tensor for output and potentially source value
        """
        if isinstance(self.tensor, TensorDescription):
            self.tensor = self.tensor_view_from_td(self.tensor)
        if isinstance(self.value, TensorDescription):
            self.value = self.tensor_view_from_td(self.value).tensor

        if self.kernel is not None:
            for index in range(len(self.params)):
                if isinstance(self.params[index], TensorDescription):
                    self.params[index] = self.pointer_from_td(self.params[index])

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

        self.flex_entry = self.tensor_view_from_td(self.tensor).flex_entry
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
        self.flex_entry = self.tensor_view_from_td(td).flex_entry
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
