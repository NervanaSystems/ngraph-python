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
from builtins import range
import atexit

from ngraph.transformers.base import Transformer, DeviceBufferStorage, DeviceBufferReference, \
    DeviceTensor
from ngraph.op_graph.op_graph import AbsoluteOneDOp, AddOneDim, AddZeroDim, Argmax, Argmin, \
    ContiguousOp, CosOneDOp, Op, \
    DivideOneDim, DivideZeroDim, DotOneDimensional, DotTwoDimensional, DotTwoByOne, \
    ModOneDim, ModZeroDim, \
    EqualOneDim, EqualZeroDim, ExpOneDOp, \
    GreaterOneDim, GreaterZeroDim, GreaterEqualOneDim, GreaterEqualZeroDim, \
    LessOneDim, LessZeroDim, \
    LessEqualOneDim, LessEqualZeroDim, LogOneDOp, Max, MaximumOneDim, MaximumZeroDim, Min, \
    MinimumOneDim, MinimumZeroDim, \
    MultiplyOneDim, MultiplyZeroDim, \
    NegativeOneDOp, NotEqualOneDim, NotEqualZeroDim, OneHotOp, Power, ReciprocalOneDOp, \
    RngOp, \
    AssignOneDOp, SignOneDOp, SinOneDOp, SqrtOneDOp, SquareOneDOp, \
    SubtractOneDim, SubtractZeroDim, \
    Sum, TanhOneDOp, TensorSizeOp, Fill, TensorDescription, \
    Function, SetItemOp
from ngraph.op_graph.convolution import ConvolutionOp, bprop_conv, update_conv
from ngraph.op_graph.pooling import PoolingOp, BpropPoolOp
# TODO: re-enable fusion
# from ngraph.analysis.fusion import gpu_fusible
from ngraph.util.generics import generic_method

from ngraph.transformers.passes.gpulayout import GPUTensorLayout

from ngraph.transformers.gpu.float_ew2 import _prepare_compound_kernel, CudaSourceFile
from ngraph.transformers.gpu.kernel import GPUKernel, pointer_from_td
from ngraph.transformers.gpu.gemm import GEMMKernel
from ngraph.transformers.gpu.conv import ConvFpropKernel, ConvBpropKernel, ConvUpdateKernel
from ngraph.transformers.gpu.pool import PoolFpropKernel, PoolBpropKernel
from ngraph.transformers.gpu.tensor_ops import DimShuffleKernel, FillKernel, SetItemKernel, \
    RngFillKernel
from ngraph.transformers.gpu.kernels.cuda.copy_transpose import _get_copy_transpose_kernel
from ngraph.transformers.gpu.util import _get_events, _get_scratch_data, _reset_scratch_data, \
    _get_sm_count, get_cache_dir

import numpy as np
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray
from pycuda.curandom import MRG32k3aRandomNumberGenerator as rng_mrg


_none_slice = slice(None, None, None)


class ElementWiseKernel(GPUKernel):
    """
    Kernel type used to execute one or more simple elementwise ops. This can
    be either a single op or a list of fused ops which corresponds to a
    Function object in the graph. In the case of regular ops (fused or single)
    the kernel generator in float_ew2 will be used to generate a CUDA C kernel
    that executes these ops. As the function is transformed by the transformer,
    we buffer ops into a list and then compile the kernel at the end.

    Some ops (non regular such as GEMM, convolution) are not handled by this kernel
    generator.

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU

    Attributes:
        ops_buffer (:obj:`list` of :obj:`tuple`): A list of operations to be
            performed by this kernel
        params (list): Parameters to pass to the compiled GPU kernel
        kernel (pycuda.driver.Function): Handle to the compiled GPU kernel
        shared_size (int): Size of shared memory needed by kernel
    """
    def __init__(self, transformer):
        super(ElementWiseKernel, self).__init__(transformer)
        self.ops_buffer = []
        self.params = None
        self.kernel = None
        self.shared_size = 0

    @generic_method(Op)
    def add_op(self, op, *args):
        if op.is_device_op:
            raise ValueError("Unhandled op: {}".format(op))

    @add_op.on_type(AbsoluteOneDOp)
    def add_op(self, op, out, x):
        self._buffer_op("abs", x=x, out=out)

    @add_op.on_type(AddOneDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("add", x=x, y=y, out=out)

    @add_op.on_type(AddZeroDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("add", x=x, y=y, out=out)

    @add_op.on_type(Argmax)
    def add_op(self, op, out, x):
        self._buffer_op("argmax",
                        x=x,
                        y=self.transformer.device_register_storage(x.dtype, None),
                        axis=0,
                        out=out)

    @add_op.on_type(Argmin)
    def add_op(self, op, out, x):
        self._buffer_op("argmin",
                        x=x,
                        y=self.transformer.device_register_storage(x.dtype, None),
                        axis=0,
                        out=out)

    @add_op.on_type(ConvolutionOp)
    def add_op(self, op, outputs, inputs, filters):
        self._buffer_op("fprop_conv", op.dims, inputs, filters, outputs)

    @add_op.on_type(bprop_conv)
    def add_op(self, op, outputs, delta, filters):
        self._buffer_op("bprop_conv", op.dims, filters, delta, outputs)

    @add_op.on_type(update_conv)
    def add_op(self, op, outputs, delta, inputs):
        self._buffer_op("update_conv", op.dims, inputs, delta, outputs)

    @add_op.on_type(PoolingOp)
    def add_op(self, op, outputs, inputs, argmax):
        self._buffer_op("fprop_pool", op.dims, inputs, outputs, argmax)

    @add_op.on_type(BpropPoolOp)
    def add_op(self, op, outputs, delta, argmax):
        self._buffer_op("bprop_pool", op.dims, delta, outputs, argmax)

    @add_op.on_type(CosOneDOp)
    def add_op(self, op, out, x):
        self._buffer_op("cos", x=x, out=out)

    @add_op.on_type(DivideOneDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("div", x=x, y=y, out=out)

    @add_op.on_type(DivideZeroDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("div", x=x, y=y, out=out)

    @add_op.on_type(ModOneDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("mod", x=x, y=y, out=out)

    @add_op.on_type(ModZeroDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("mod", x=x, y=y, out=out)

    @add_op.on_type(DotOneDimensional)
    def add_op(self, op, out, x, y):
        self._buffer_op("dot", x=x, y=y, out=out)

    @add_op.on_type(DotTwoDimensional)
    def add_op(self, op, out, x, y):
        self._buffer_op("dot", x=x, y=y, out=out)

    @add_op.on_type(DotTwoByOne)
    def add_op(self, op, out, x, y):
        self._buffer_op("dot", x=x, y=y, out=out)

    @add_op.on_type(EqualOneDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("eq", x=x, y=y, out=out)

    @add_op.on_type(EqualZeroDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("eq", x=x, y=y, out=out)

    @add_op.on_type(ExpOneDOp)
    def add_op(self, op, out, x):
        self._buffer_op("exp", x=x, out=out)

    @add_op.on_type(GreaterOneDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("gt", x=x, y=y, out=out)

    @add_op.on_type(GreaterZeroDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("gt", x=x, y=y, out=out)

    @add_op.on_type(GreaterEqualOneDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("ge", x=x, y=y, out=out)

    @add_op.on_type(GreaterEqualZeroDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("ge", x=x, y=y, out=out)

    @add_op.on_type(LessOneDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("lt", x=x, y=y, out=out)

    @add_op.on_type(LessZeroDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("lt", x=x, y=y, out=out)

    @add_op.on_type(LessEqualOneDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("le", x=x, y=y, out=out)

    @add_op.on_type(LessEqualZeroDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("le", x=x, y=y, out=out)

    @add_op.on_type(LogOneDOp)
    def add_op(self, op, out, x):
        self._buffer_op("log", x=x, out=out)

    @add_op.on_type(Max)
    def add_op(self, op, out, x):
        self._buffer_op("max", x=x, axis=0, out=out)

    @add_op.on_type(MaximumOneDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("maximum", x=x, y=y, out=out)

    @add_op.on_type(MaximumZeroDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("maximum", x=x, y=y, out=out)

    @add_op.on_type(Min)
    def add_op(self, op, out, x):
        self._buffer_op("min", x=x, axis=0, out=out)

    @add_op.on_type(MinimumOneDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("minimum", x=x, y=y, out=out)

    @add_op.on_type(MinimumZeroDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("minimum", x=x, y=y, out=out)

    @add_op.on_type(MultiplyOneDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("mul", x=x, y=y, out=out)

    @add_op.on_type(MultiplyZeroDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("mul", x=x, y=y, out=out)

    @add_op.on_type(NegativeOneDOp)
    def add_op(self, op, out, x):
        self._buffer_op("neg", x=x, out=out)

    @add_op.on_type(NotEqualOneDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("ne", x=x, y=y, out=out)

    @add_op.on_type(NotEqualZeroDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("ne", x=x, y=y, out=out)

    @add_op.on_type(OneHotOp)
    def add_op(self, op, out, x):
        self._buffer_op("onehot", x=x, out=out)

    @add_op.on_type(Power)
    def add_op(self, op, out, x, y):
        self._buffer_op("pow", x=x, y=y, out=out)

    @add_op.on_type(ReciprocalOneDOp)
    def add_op(self, op, out, x):
        self._buffer_op("rcp", x=x, out=out)

    @add_op.on_type(AssignOneDOp)
    def add_op(self, op, out, tensor, value):
        self._buffer_op("assign", x=value, out=tensor)

    @add_op.on_type(SignOneDOp)
    def add_op(self, op, out, x):
        self._buffer_op("sgn", x=x, out=out)

    @add_op.on_type(SinOneDOp)
    def add_op(self, op, out, x):
        self._buffer_op("sin", x=x, out=out)

    @add_op.on_type(SqrtOneDOp)
    def add_op(self, op, out, x):
        self._buffer_op("sqrt", x=x, out=out)

    @add_op.on_type(SquareOneDOp)
    def add_op(self, op, out, x):
        self._buffer_op("sqr", x=x, out=out)

    @add_op.on_type(SubtractOneDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("sub", x=x, y=y, out=out)

    @add_op.on_type(SubtractZeroDim)
    def add_op(self, op, out, x, y):
        self._buffer_op("sub", x=x, y=y, out=out)

    @add_op.on_type(Sum)
    def add_op(self, op, out, x):
        self._buffer_op("sum", x=x, axis=0, out=out)

    @add_op.on_type(TanhOneDOp)
    def add_op(self, op, out, x):
        self._buffer_op("tanh", x=x, out=out)

    def _buffer_op(self, op, x=None, y=None, out=None, axis=None, extra=None):
        """
        Adds an op to the list of ops to be compiled into a kernel

        Arguments:
            op (string): Name of the op
            x (TensorDescription): TensorDescription for input 0
            y (TensorDescription): TensorDescription for input 1
            out (TensorDescription): Tensor description for output
            axis (int): For reduction ops, indicate the axis to reduce
                along
        """
        self.ops_buffer.append((op, x, y, out, axis, extra))

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

        super(ElementWiseKernel, self).bind_buffers()

    def generate_source(self, sourcefile=None):
        """
        Generates source code and adds it to a kernel file to be compiled later.
        First checks if this is a compound kernel which needs to be compiled.
        In cases where only a single special op are contained (dot, dimshuffle, etc)
        there is no compounding and the NervanaGPU implementation is called directly
        at run time.

        Arguments:
            sourcefile (CudaSourceFile): Object handling cuda source file generation
        """
        if len(self.ops_buffer) == 0:
            return False

        if sourcefile is not None:
            # Code generation and compilation are only separate when a sourcefile is
            # provided
            self.name, self.params = sourcefile.add_kernel(self.ops_buffer)

        return True

    def compile(self, sourcefile=None):
        """
        Compiles ops buffer into a GPU kernel.
        """
        if len(self.ops_buffer) == 0:
            return False

        if sourcefile is None:
            # Generate and compile single kernel
            self.kernel, self.params, self.shared_size = \
                _prepare_compound_kernel(self.ops_buffer)
        else:
            # Get kernel object from compiled sourcefile
            self.kernel = sourcefile.get_kernel(self.name)

        return True

    def execute(self):
        self.kernel.prepared_async_call(*self.params,
                                        shared_size=self.shared_size)


class GPUKernelGroup(object):
    """
    A group of GPU kernels which corresponds to a Computation object. Since we
    can't always compound all ops from a Computation into a single GPU kernel,
    this object provides a container for multiple kernels. The class implements
    __call__ which is used to execute the kernel group at evaluation time.

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU
        kernels (:obj:`list` of :class:`GPUKernel`): List of compiled GPUKernel
            objects to run at evaluation time

    Attributes:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU
        ng (NervanaGPU): Neon backend used to execute special ops
        kernels (:obj:`list` of :class:`GPUKernel`): List of compiled GPUKernel
            objects to run at evaluation time
        sourcefile (CudaSourceFile): CUDA C source file that will contain all
            GPU elementwise kernels for this computation
    """

    def __init__(self, transformer, name):
        self.transformer = transformer
        self.kernels = []
        self.name = name
        self.sourcefile = CudaSourceFile(name)

    @generic_method(Op)
    def add_kernel(self, op):
        # Use default kernel generator for single operation
        out = op.tensor_description()
        call_info = (_ for _ in op.call_info())

        kernel = ElementWiseKernel(self.transformer)
        kernel.add_op(op, out, *call_info)

        if kernel.generate_source(self.sourcefile):
            self.kernels.append(kernel)

    @add_kernel.on_type(Function)
    def add_kernel(self, op):
        # Iterate over compounded operations and build kernel for them
        kernel = ElementWiseKernel(self.transformer)
        for sub_op in op.instructions:
            out = sub_op.tensor_description()
            call_info = (_ for _ in sub_op.call_info())
            kernel.add_op(sub_op, out, *call_info)

        if kernel.generate_source(self.sourcefile):
            self.kernels.append(kernel)

    @add_kernel.on_type(ConvolutionOp)
    def add_kernel(self, op):
        self.kernels.append(ConvFpropKernel(self.transformer, op))

    @add_kernel.on_type(bprop_conv)
    def add_kernel(self, op):
        self.kernels.append(ConvBpropKernel(self.transformer, op))

    @add_kernel.on_type(update_conv)
    def add_kernel(self, op):
        self.kernels.append(ConvUpdateKernel(self.transformer, op))

    @add_kernel.on_type(DotOneDimensional)
    def add_kernel(self, op):
        self.kernels.append(GEMMKernel(self.transformer, op))

    @add_kernel.on_type(DotTwoDimensional)
    def add_kernel(self, op):
        self.kernels.append(GEMMKernel(self.transformer, op))

    @add_kernel.on_type(DotTwoByOne)
    def add_kernel(self, op):
        self.kernels.append(GEMMKernel(self.transformer, op))

    @add_kernel.on_type(ContiguousOp)
    def add_kernel(self, op):
        self.kernels.append(DimShuffleKernel(self.transformer, op))

    @add_kernel.on_type(Fill)
    def add_kernel(self, op):
        self.kernels.append(FillKernel(self.transformer, op.call_info()[0], op.scalar))

    @add_kernel.on_type(RngOp)
    def add_kernel(self, op):
        self.kernels.append(RngFillKernel(self.transformer,
                                          op.tensor_description(),
                                          op.distribution,
                                          op.params))

    @add_kernel.on_type(PoolingOp)
    def add_kernel(self, op):
        self.kernels.append(PoolFpropKernel(self.transformer, op))

    @add_kernel.on_type(BpropPoolOp)
    def add_kernel(self, op):
        self.kernels.append(PoolBpropKernel(self.transformer, op))

    @add_kernel.on_type(SetItemOp)
    def add_kernel(self, op):
        self.kernels.append(SetItemKernel(self.transformer, op))

    @add_kernel.on_type(TensorSizeOp)
    def add_kernel(self, op):
        self.kernels.append(FillKernel(self.transformer, op.tensor_description(),
                                       op.reduction_axes.size))

    def compile_all(self):
        """
        Compiles all CUDA C kernels in this group's source file and updates the
        corresponding kernel objects with their pycuda prepared functions.
        """
        self.sourcefile.compile()
        for kernel in self.kernels:
            kernel.compile(self.sourcefile)

    def setup_kernel_execute(self, kernel):
        pass

    def __call__(self):
        """
        Loops over all kernels contained in this group and executes them. Since buffers
        are allocated between kernel compilation and the first run of the computation,
        we have to update GPU memory pointers in the arguments for each kernel on the
        first execution of each kernel.
        """
        for k in self.kernels:
            if not k.buffers_bound:
                k.bind_buffers()

            self.setup_kernel_execute(k)
            k.execute()


class GPUBufferAllocator():
    """
    Class responsible for allocating a buffer in GPU memory and calling
    allocators for tensor views of that buffer. The class implements __call__
    which is used to perform allocation.

    Arguments:
        dev_buffer (GPUDeviceBufferStorage): Device storage object to be
            allocated

    Attributes:
        bytes (int): Size of buffer to allocate
        view_allocators (:obj:`list` of :class:`GPUTensorAllocator`): List of
            allocators using this buffer for storage
        _buffer (pycuda.driver.DeviceAllocation): Device memory handle
    """

    def __init__(self, dev_buffer):
        self.bytes = dev_buffer.bytes
        self.view_allocators = []
        self._buffer = None

    def __call__(self):
        """
        Allocate the device memory buffer then loop over tensors which use the
        buffer and call their allocators to create views
        """
        self._buffer = drv.mem_alloc(self.bytes)
        for view_alloc in self.view_allocators:
            view_alloc(self._buffer)

    def add_view_allocator(self, view_alloc):
        """
        Add reference to an allocator for a tensor view of this buffer

        Arguments:
            view_alloc (GPUTensorAllocator): Tensor allocator which uses this
                buffer
        """
        self.view_allocators.append(view_alloc)


class GPUTensorAllocator():
    """
    Class responsible for allocating a tensor view of a device memory buffer.
    The class implements __call__ which creates a neon GPUTensor bound to the
    specified device allocation

    Arguments:
        tensor (GPUDeviceTensor): Tensor to allocate
        transformer (GPUTransformer): GPUTransformer containing a NervanaGPU
            which is used as the backend for the GPUTensor

    Attributes:
        transformer (GPUTransformer): GPUTransformer containing a NervanaGPU
            which is used as the backend for the GPUTensor
        tensor_name (string): Name of the tensor used in GPUTransformer dict to
            store the allocated tensor
        tensor_description (TensorDescription): Description of the view
        _tensor (GPUTensor): Allocated neon GPUTensor
    """

    def __init__(self, tensor, transformer):
        self.transformer = transformer
        self.tensor_name = tensor.name
        self.tensor_description = tensor.tensor_description
        self._tensor = None
        # TODO: this doesn't feel super clean, but just to get something working...
        if hasattr(transformer, 'use_flex_dtype'):
            self.dtype = np.int16
        else:
            self.dtype = self.tensor_description.dtype

    def __call__(self, buffer_alloc):
        """
        Allocates the GPUTensor object as a view of a pre-allocated buffer.

        Arguments:
            buffer_alloc (DeviceAllocation): Memory handle returned by pycuda
                allocator
        """
        tensor_description = self.tensor_description

        gpudata = int(buffer_alloc) + tensor_description.offset
        new_tensor = GPUArray(tensor_description.shape,
                              self.dtype,
                              gpudata=gpudata,
                              strides=tensor_description.strides)

        self._tensor = new_tensor
        self.transformer.tensors[self.tensor_name] = self._tensor


class GPURegister():
    """
    Object representing a register in a GPU kernel used to store the result of
    an intermediate computation which does not need to be written to a buffer

    Arguments:
        dtype (dtype): Variable type of the register
        name (string): Name of the register
    """

    def __init__(self, dtype, name):
        self.dtype = dtype
        self.name = name


class GPUDeviceBufferReference(DeviceBufferReference):
    """
    Analogous to NumPyDeviceBufferReference.
    """
    def __init__(self, transformer, **kwargs):
        super(GPUDeviceBufferReference, self).__init__(transformer, **kwargs)


class GPUDeviceBufferStorage(DeviceBufferStorage):
    """
    Used to transform device allocations. Analogous to NumPyDeviceBufferStorage.
    """

    def __init__(self, transformer, bytes, dtype, **kwargs):
        super(GPUDeviceBufferStorage, self).__init__(transformer, bytes, dtype, **kwargs)
        self.storage = None

    def create_device_tensor(self, tensor_description):
        shape_str = "_".join((str(_) for _ in tensor_description.shape))
        return GPUDeviceTensor(self.transformer, self, tensor_description,
                               name="v_" + tensor_description.name + "_" + shape_str)

    @property
    def ref_str(self):
        """
        :return: name to reference variable.
        """
        return self.name

    def transform_allocate(self):
        buffer_alloc = GPUBufferAllocator(self)
        self.transformer.buffer_allocators.append(buffer_alloc)

        # Allocate all views of this buffer
        self.transformer.current_buffer = buffer_alloc
        self.transform_allocate_views()
        self.transformer.current_buffer = None


class GPUDeviceTensor(DeviceTensor):
    """
    Used to transform device tensor allocations. Analogous to NumPyDeviceTensor.
    """
    def __init__(self, transformer, device_buffer, tensor_description, **kwargs):
        super(GPUDeviceTensor, self).__init__(transformer, device_buffer, tensor_description,
                                              **kwargs)
        self.__tensor = None

    @property
    def tensor(self):
        if self.__tensor is None:
            self.__tensor = self.transformer.tensors[self.name]
        return self.__tensor

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def ref_str(self):
        """
        :return: name to reference variable.
        """
        return self.name

    def transform_allocate(self):
        tensor_alloc = GPUTensorAllocator(self, self.transformer)
        self.transformer.add_view_allocator(tensor_alloc)

    def get(self, tensor):
        """
        Copy the device tensor to a numpy array.

        Arguments:
            tensor (np.ndarray): Optional output array

        Returns:
            Numpy array containing tensor data
        """

        if np.sum(self.tensor.strides) != 0:
            if self.is_contiguous or self.tensor.shape == () or np.prod(self.tensor.shape) == 1:
                contig_tensor = self.tensor
            else:
                # Need to do memcpy from contiguous device memory
                contig_tensor = self.as_contiguous()

            if tensor is None:
                return contig_tensor.get()
            tensor[:] = contig_tensor.get()
        else:
            # Tensor is just a broadcasted scalar, get scalar value and fill output array
            view = GPUArray((1, ), dtype=self.tensor.dtype, gpudata=self.tensor.gpudata)[0]
            value = view.get()

            if tensor is None:
                out = np.ndarray(self.tensor.shape, dtype=self.tensor.dtype)
                out.fill(value)
                return out
            tensor.fill(value)

        return tensor

    def set_dtype(self, dtype):
        self.tensor_description.dtype = dtype

    def __getitem__(self, index):
        if index is None or index == _none_slice or index == ():
            return self.tensor
        elif not isinstance(index, tuple):
            index = (index,)

        # Slice tensor by changing shape, strides, and base address
        new_shape = []
        new_offset = 0
        new_strides = []
        seen_ellipsis = False

        shape = self.tensor.shape
        dtype = self.tensor.dtype
        strides = self.tensor.strides

        # Iterate over axes of index to compute new offset, shape, strides
        array_axis = 0
        for index_axis in range(len(index)):
            index_entry = index[index_axis]

            if array_axis > len(shape):
                raise IndexError("Too many axes in index")

            if isinstance(index_entry, slice):
                # Standard slicing (start:stop:step)
                start, stop, idx_strides = index_entry.indices(shape[array_axis])

                new_offset += (start * strides[array_axis])
                new_shape.append(-((start - stop) // idx_strides))
                new_strides.append(idx_strides * strides[array_axis])

                array_axis += 1
            elif isinstance(index_entry, (int, np.integer)):
                # Single index value
                new_offset += (index_entry * strides[array_axis])
                array_axis += 1
            elif index_entry is Ellipsis:
                # Use same shape as original for these axes
                if seen_ellipsis:
                    raise IndexError(
                        "More than one ellipsis not allowed in index")
                seen_ellipsis = True

                remaining_index_count = len(index) - (index_axis + 1)
                new_array_axis = len(shape) - remaining_index_count
                if new_array_axis < array_axis:
                    raise IndexError("Invalid use of ellipsis in index")
                while array_axis < new_array_axis:
                    new_shape.append(shape[array_axis])
                    new_strides.append(strides[array_axis])
                    array_axis += 1
            else:
                raise IndexError("Invalid subindex %s in axis %d" % (index_entry, index_axis))

        # Create view
        return GPUArray(new_shape,
                        dtype,
                        strides=new_strides,
                        gpudata=(self.tensor.gpudata + new_offset))

    def __setitem__(self, key, value):
        sliced = self.__getitem__(key)

        # Use fill for scalar values
        if type(value) == np.float32 or type(value) == np.float64 or \
                type(value) == float:
            sliced.fill(value)
        elif type(value) == np.int32 or type(value) == np.int64 or \
                type(value) == int:
            sliced.fill(value)
        elif self.tensor.shape == () or np.prod(self.tensor.shape) == 1:
            sliced.fill(value)
        elif np.sum(self.tensor.strides) == 0:
            view = GPUArray((1, ), dtype=self.tensor.dtype)
            view.fill(value)
        else:
            # Convert to correct dtype if necessary
            if value.dtype != self.tensor.dtype:
                new_value = np.ndarray(self.tensor.shape, dtype=self.tensor.dtype)
                new_value[:] = value
                value = new_value

            # Reshape to satisfy pycuda if necessary
            if sliced.shape != value.shape:
                sliced = self.tensor.reshape(value.shape)

            if self.is_contiguous and self.strides_contiguous(value):
                if sliced.shape == ():
                    sliced.reshape((1,))[:] = value.reshape((1,))
                else:
                    sliced[:] = value
            elif type(value) == GPUArray:
                self.from_other(value, sliced)
            else:
                contig_tensor = GPUArray(value.shape, self.tensor.dtype)
                contig_tensor[:] = value
                self.from_other(contig_tensor, sliced)

    @property
    def is_contiguous(self):
        return self.strides_contiguous()

    def strides_contiguous(self, tensor=None):
        if tensor is None:
            tensor = self.tensor

        if tensor.shape == ():
            return True

        # Compute contiguous strides and compare with tensor strides
        contiguous_strides = [tensor.dtype.itemsize]
        for dim in reversed(tensor.shape[1:]):
            contiguous_strides.insert(0, contiguous_strides[0] * dim)
        return (tuple(contiguous_strides) == tensor.strides)

    def as_contiguous(self):
        """
        Creates a new GPUArray with the same dimensions, but using contiguous memory

        Returns:
            New contiguous GPUArray with separate underlying device allocation
        """
        contig_tensor = GPUArray(self.tensor.shape, dtype=self.tensor.dtype)
        src_strides = [s // self.tensor.dtype.itemsize for s in self.tensor.strides]
        dst_strides = [s // contig_tensor.dtype.itemsize for s in contig_tensor.strides]
        kernel = _get_copy_transpose_kernel(self.tensor.dtype,
                                            self.tensor.shape,
                                            range(len(self.tensor.shape)))
        params = [contig_tensor.gpudata, self.tensor.gpudata] + list(kernel.args)
        params = params + src_strides + dst_strides
        kernel.prepared_async_call(kernel.grid, kernel.block, None, *params)
        return contig_tensor

    def from_other(self, tensor, dest=None):
        """
        Copies from another GPUArray with the same dimensions into this tensor. Handles
        discontiguous strides.

        Arguments:
            tensor (GPUArray): Contiguous tensor with same dimensions to use as source
        """
        if dest is None:
            dest = self.tensor

        src_strides = [s // tensor.dtype.itemsize for s in tensor.strides]
        dst_strides = [s // dest.dtype.itemsize for s in dest.strides]
        kernel = _get_copy_transpose_kernel(tensor.dtype,
                                            tensor.shape,
                                            range(len(tensor.shape)))
        params = [dest.gpudata, tensor.gpudata] + list(kernel.args)
        params = params + src_strides + dst_strides
        kernel.prepared_async_call(kernel.grid, kernel.block, None, *params)


class GPURuntime(object):
    def __init__(self, device_id=None, enable_winograd=True, deterministic=True,
                 scratch_size=0):
        drv.init()
        self.device_id = device_id if device_id is not None else 0

        # check compute capability
        self.compute_capability = drv.Device(self.device_id).compute_capability()
        if self.compute_capability[0] < 3:
            raise RuntimeError("Unsupported GPU")

        # context
        self.ctx = drv.Device(self.device_id).make_context()

        # attributes
        self.stream = None
        self.warmup = False
        self.scratch_size = scratch_size
        self.scratch_offset = 0
        self.sm_count = _get_sm_count()

        # store GPU memory size in bytes
        self.gpu_memory_size = drv.mem_get_info()[1]

        # Fall back to CUDA C kernels on older (pre-Maxwell) GPU generations
        if self.compute_capability[0] < 5:
            # TODO: this is not fully supported in graph yet
            self.use_cudac_kernels = True
        else:
            self.use_cudac_kernels = False

        # TODO
        # self.cublas_handle = cublas.cublasCreate()
        self.pcg = rng_mrg()

        self.enable_winograd = enable_winograd
        self.deterministic = deterministic
        self.cache_dir = get_cache_dir()

    def cleanup(self):
        try:
            self.ctx.pop()
            self.ctx.detach()
        except drv.Error:
            pass

    def get_events(self):
        return _get_events()

    def scratch_buffer_reset(self):
        self.scratch_size = 0
        self.scratch_offset = 0
        _reset_scratch_data()

    def scratch_buffer_init(self):
        self.scratch_offset = 0

    def scratch_buffer(self, size):

        if size & 127 != 0:
            size += 128 - (size & 127)

        if size > self.scratch_size:
            raise RuntimeError(
                "nervanagpu.scratch_size(%d) is too small for this operation(%d)" % (
                    self.scratch_size, size))

        self.scratch_offset = size

        return int(_get_scratch_data(self.scratch_size))

    def scratch_buffer_offset(self, size):

        if size & 127 != 0:
            size += 128 - (size & 127)

        if size + self.scratch_offset > self.scratch_size:
            raise RuntimeError(
                "nervanagpu.scratch_size(%d) is too small for this operation(%d, %d)" % (
                    self.scratch_size, size, self.scratch_offset))

        data = int(_get_scratch_data(self.scratch_size)) + self.scratch_offset
        self.scratch_offset += size

        return data

    def set_scratch_size(self, *args):

        total_size = 0
        for size in args:
            if size & 127 != 0:
                size += 128 - (size & 127)
            total_size += size

        if total_size > self.scratch_size:
            self.scratch_size = total_size


class GPUTransformer(Transformer):
    """
    Transformer for executing graphs on a GPU, backed by pycuda and NervanaGPU.

    Given a list of ops you want to compute the results of, this transformer
    will generate allocators and kernels to execute the graph on a GPU.
    """
    __runtime = None

    transformer_name = "gpu"

    @staticmethod
    def close_gpu():
        if GPUTransformer.__runtime is not None:
            GPUTransformer.__runtime.cleanup()
            GPUTransformer.__runtime = None

    def __init__(self, **kwargs):
        # TODO: Re-enable fusion
        # super(GPUTransformer, self).__init__(fusion=gpu_fusible, **kwargs)
        super(GPUTransformer, self).__init__(**kwargs)

        self.graph_passes.insert(0, GPUTensorLayout())

        self.buffer_allocators = []
        self.kernel_groups = dict()
        self.tensors = dict()
        self.argmax_tensors = dict()
        self.finished_transform = False
        self.current_buffer = None

        if GPUTransformer.__runtime is None:
            GPUTransformer.__runtime = GPURuntime()
            atexit.register(GPUTransformer.close_gpu)

        self.runtime = GPUTransformer.__runtime

    def device_register_storage(self, dtype, name):
        return GPURegister(dtype, name)

    def device_buffer_storage(self, bytes, dtype, name):
        """
        Make a DeviceBuffer.

        Arguments:
            bytes: Size of buffer.
            alignment: Alignment of buffer.

        Returns: A DeviceBuffer.
        """
        return GPUDeviceBufferStorage(self, bytes, dtype, name="a_" + name)

    def device_buffer_reference(self):
        """
        Make a DeviceBufferReference.

        Returns: A DeviceBufferReference.
        """
        return GPUDeviceBufferReference(self)

    def add_view_allocator(self, view_alloc):
        self.current_buffer.add_view_allocator(view_alloc)

    def start_transform_allocate(self):
        pass

    def finish_transform_allocate(self):
        pass

    def gpu_kernel_group(self, name):
        return GPUKernelGroup(self, name)

    def transform_ordered_ops(self, ordered_ops, name):
        # Create kernel group
        kernel_group = self.gpu_kernel_group(name)
        for fun in ordered_ops:
            kernel_group.add_kernel(fun)

        kernel_group.compile_all()
        self.kernel_groups[name] = kernel_group

        return name

    def finish_transform(self):
        if self.finished_transform:
            return

        for computation in self.computations:
            executor = self.kernel_groups[computation.name]
            computation.executor = executor

        self.finished_transform = True

    def allocate_storage(self):
        for alloc in self.buffer_allocators:
            alloc()
