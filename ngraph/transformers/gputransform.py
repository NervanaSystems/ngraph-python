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

from neon.backends.nervanagpu import GPUTensor
from neon import NervanaObject
from neon.backends import gen_backend

from ngraph.transformers.base import Transformer, DeviceBufferStorage, DeviceBufferReference, \
    DeviceTensor
from ngraph.op_graph.op_graph import AbsoluteOneDOp, AddOneDim, AddZeroDim, Argmax, Argmin, \
    CosOneDOp, Op, \
    DivideOneDim, DivideZeroDim, DotOneDimensional, DotTwoDimensional, DotTwoByOne, \
    ModOneDim, ModZeroDim, \
    EqualOneDim, EqualZeroDim, ExpOneDOp, \
    GreaterOneDim, GreaterZeroDim, GreaterEqualOneDim, GreaterEqualZeroDim, \
    LessOneDim, LessZeroDim, \
    LessEqualOneDim, LessEqualZeroDim, LogOneDOp, Max, MaximumOneDim, MaximumZeroDim, Min, \
    MinimumOneDim, MinimumZeroDim, \
    MultiplyOneDim, MultiplyZeroDim, \
    NegativeOneDOp, NotEqualOneDim, NotEqualZeroDim, OneHotOp, Power, ReciprocalOneDOp, \
    AssignOneDOp, SignOneDOp, SinOneDOp, SqrtOneDOp, SquareOneDOp, \
    SubtractOneDim, SubtractZeroDim, \
    Sum, TanhOneDOp, TensorSizeOp, Fill, TensorDescription, Unslice, Stack, Dimshuffle, \
    Function, SetItemOneDOp
from ngraph.op_graph.convolution import ConvolutionOp, bprop_conv, update_conv
from ngraph.op_graph.pooling import PoolingOp, BpropPoolOp
# TODO: re-enable fusion
# from ngraph.analysis.fusion import gpu_fusible
from ngraph.util.generics import generic_method

from ngraph.transformers.gpu.float_ew2 import _prepare_compound_kernel, CudaSourceFile
from ngraph.transformers.gpu.kernel import GPUKernel, pointer_from_td
from ngraph.transformers.gpu.gemm import GEMMKernel
from ngraph.transformers.gpu.conv import ConvFpropKernel, ConvBpropKernel, ConvUpdateKernel
from ngraph.transformers.gpu.pool import PoolFpropKernel, PoolBpropKernel
from ngraph.transformers.gpu.tensor_ops import DimShuffleKernel, FillKernel, SetItemKernel, \
    UnsliceKernel

import numpy as np
import pycuda.driver as drv


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

    @add_op.on_type(Stack)
    def add_op(self, op, out, *args):
        # TODO: we may want to have the inputs write into slices of a
        # preallocated buffer for this op.
        # We cannot use the numpy stack function as it is unavailable in
        # older versions.
        # self.append("o={}", out)
        # slices = [slice(None)] * len(op.axes)
        # for i, arg in enumerate(args):
        #    slices[op.pos] = i
        #    self.append("o.__setitem__({s}, {x})", s=tuple(slices), x=arg)
        raise ValueError("Unhandled op: {}".format(op))

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


class GPUKernelGroup():
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
    """

    def __init__(self, transformer, name):
        self.transformer = transformer
        self.ng = transformer.ng
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

    @add_kernel.on_type(Dimshuffle)
    def add_kernel(self, op):
        self.kernels.append(DimShuffleKernel(self.transformer, op))

    @add_kernel.on_type(Fill)
    def add_kernel(self, op):
        self.kernels.append(FillKernel(self.transformer, op.tensor_description(), op.scalar))

    @add_kernel.on_type(PoolingOp)
    def add_kernel(self, op):
        self.kernels.append(PoolFpropKernel(self.transformer, op))

    @add_kernel.on_type(BpropPoolOp)
    def add_kernel(self, op):
        self.kernels.append(PoolBpropKernel(self.transformer, op))

    @add_kernel.on_type(SetItemOneDOp)
    def add_kernel(self, op):
        self.kernels.append(SetItemKernel(self.transformer, op))

    @add_kernel.on_type(TensorSizeOp)
    def add_kernel(self, op):
        self.kernels.append(FillKernel(self.transformer, op.tensor_description(),
                                       op.reduction_axes.size))

    @add_kernel.on_type(Unslice)
    def add_kernel(self, op):
        self.kernels.append(UnsliceKernel(self.transformer, op))

    def compile_all(self):
        self.sourcefile.compile()
        for kernel in self.kernels:
            kernel.compile(self.sourcefile)

    def __call__(self):
        for k in self.kernels:
            if not k.buffers_bound:
                k.bind_buffers()

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

    def __call__(self, buffer_alloc):
        """
        Allocates the GPUTensor object as a view of a pre-allocated buffer.

        Arguments:
            buffer_alloc (DeviceAllocation): Memory handle returned by pycuda
                allocator
        """
        tensor_description = self.tensor_description

        if tensor_description.shape == ():
            shape = (1, )
        else:
            shape = tensor_description.shape

        if tensor_description.strides == ():
            strides = (1, )
        else:
            # Note that TensorDescription strides are in units of bytes, but
            # GPUTensor expects units of elements
            strides = [s // tensor_description.dtype.itemsize for s in tensor_description.strides]
            strides = tuple(strides)

        if len(shape) == 1 and len(strides) == 1:
            shape = (shape[0], 1)
            strides = (strides[0], 0)

        gpudata = int(buffer_alloc) + tensor_description.offset
        new_tensor = GPUTensor(self.transformer.ng,
                               shape,
                               dtype=tensor_description.dtype,
                               gpudata=gpudata,
                               strides=strides)

        if new_tensor.strides[0] < new_tensor.strides[-1]:
            new_tensor.is_trans = True

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


class GPUDeviceBufferReference(DeviceBufferReference):
    """
    Analogous to NumPyDeviceBufferReference.
    """
    def __init__(self, transformer, **kwargs):
        super(GPUDeviceBufferReference, self).__init__(transformer, **kwargs)


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
    def ref_str(self):
        """
        :return: name to reference variable.
        """
        return self.name

    def transform_allocate(self):
        tensor_alloc = GPUTensorAllocator(self, self.transformer)
        self.transformer.add_view_allocator(tensor_alloc)

    def get(self, tensor):
        if self.tensor.is_contiguous or (len(self.tensor.shape) == 2 and
                                         (self.tensor.shape[0] == 1 or
                                          self.tensor.shape[1] == 1)):
            np_ary = self.tensor.get().reshape(self.tensor_description.shape)
        else:
            temp_gpu_tensor = self.transformer.ng.empty(shape=self.tensor.shape,
                                                        dtype=self.tensor.dtype)
            self.transformer.ng.copy_transpose(self.tensor,
                                               temp_gpu_tensor,
                                               axes=range(len(self.tensor.shape)))
            np_ary = temp_gpu_tensor.get().reshape(self.tensor_description.shape)

        if tensor is None:
            return np_ary
        tensor[:] = np_ary

    def __getitem__(self, key):
        return self.tensor.__getitem__(key)

    def __setitem__(self, key, value):
        if type(value) == np.float32 or type(value) == np.float64:
            value = float(value)
        elif type(value) == np.int32 or type(value) == np.int64:
            value = int(value)

        if self.tensor.is_contiguous:
            self.tensor.__setitem__(key, value)
        else:
            if type(value) == np.ndarray:
                # TODO: warn?
                value = self.transformer.ng.array(value)

                if value.T.shape == self.__getitem__(key).shape:
                    self.__getitem__(key)._assign(value.T)
                else:
                    self.__getitem__(key)._assign(value)
            else:
                self.__getitem__(key)._assign(value)

    def reshape(self, shape):
        """Temporary for conv"""
        # TODO Remove when CONV is finished
        return self.tensor.reshape(shape)


class GPUTransformer(Transformer):
    """
    Transformer for executing graphs on a GPU, backed by pycuda and NervanaGPU.

    Given a list of ops you want to compute the results of, this transformer
    will generate allocators and kernels to execute the graph on a GPU.
    """
    __nervanagpu = None

    transformer_name = "gpu"

    @staticmethod
    def close_gpu():
        if GPUTransformer.__nervanagpu is not None:
            GPUTransformer.__nervanagpu.cleanup_backend()
            GPUTransformer.__nervanagpu = None

    def __init__(self, **kwargs):
        if NervanaObject.be is None or NervanaObject.be.device_type != 1:
            # This creates a backend for unit tests.
            NervanaObject.be = gen_backend('gpu')
        # TODO: Re-enable fusion
        # super(GPUTransformer, self).__init__(fusion=gpu_fusible, **kwargs)
        super(GPUTransformer, self).__init__(**kwargs)

        self.buffer_allocators = []
        self.kernel_groups = dict()
        self.tensors = dict()
        self.argmax_tensors = dict()
        self.finished_transform = False
        self.current_buffer = None
        self.closed = False

        if GPUTransformer.__nervanagpu is None:
            GPUTransformer.__nervanagpu = NervanaObject.be
            atexit.register(GPUTransformer.close_gpu)

        self.ng = GPUTransformer.__nervanagpu

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

    def transform_ordered_ops(self, ordered_ops, name):
        # Create kernel group
        kernel_group = GPUKernelGroup(self, name)
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
