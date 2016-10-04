from builtins import object, range
import atexit

from neon.backends.nervanagpu import NervanaGPU, GPUTensor

from ngraph.transformers.base import Transformer, DeviceBufferStorage, DeviceBufferReference, \
    DeviceTensor
from ngraph.op_graph.op_graph import absolute, Add, Argmax, Argmin, cos, Divide, Equal, exp, \
    Greater, GreaterEqual, Less, LessEqual, log, Max, Maximum, Min, Minimum, Multiply, \
    negative, NotEqual, onehot, Power, reciprocal, SetItem, sign, sin, sqrt, square, Subtract, \
    Sum, tanh, tensor_size, Fill, Unslice, Stack, Dot, Dimshuffle, Function
from ngraph.op_graph.convolution import convolution1d
# TODO: re-enable fusion
# from ngraph.analysis.fusion import gpu_fusible
from ngraph.util.generics import generic_method
from ngraph.op_graph.arrayaxes import TensorDescription
from ngraph.transformers.gpu.float_ew2 import _prepare_compound_kernel

import numpy as np
import pycuda.driver as drv


_none_slice = slice(None, None, None)


class GPUKernel():
    """
    Object which represents a single kernel that will run on the GPU. This can
    be either a single op or a list of fused ops which corresponds to a
    Function object in the graph. In the case of regular ops (fused or single)
    the kernel generator in float_ew2 will be used to generate a CUDA C kernel
    that executes these ops. As the function is transformed by the transformer,
    we buffer ops into a list and then compile the kernel at the end.

    Some ops (non regular) are not handled by the kernel generator and instead
    rely on neon NervanaGPU implementations. These are genrally ops which
    cannot be fused or done in place such as dot and dimshuffle. In these cases
    self.compound will be set to False.

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU

    Attributes:
        ops_buffer (:obj:`list` of :obj:`tuple`): A list of operations to be
            performed by this kernel
        params (list): Parameters to pass to the compiled GPU kernel
        kernel (pycuda.driver.Function): Handle to the compiled GPU kernel
        shared_size (int): Size of shared memory needed by kernel
        compound (bool): True if the kernel needs to be pre-compiled using the
            float_ew2 module
        buffers_bound (bool): Flag indicates if GPU addresses have been bound
            to kernel parameters
        transformer (GPUTransformer): GPU transformer containing NervanaGPU
            object which is used for ops such as dot, dimshuffle, etc.
    """

    transformer_name = "gpu"

    def __init__(self, transformer):
        self.ops_buffer = []
        self.params = None
        self.kernel = None
        self.shared_size = 0
        self.compound = True
        self.buffers_bound = False
        self.transformer = transformer

    @generic_method
    def add_op(self, op, *args):
        if op.device_op:
            raise ValueError("Unhandled op: {}".format(op))

    @add_op.on_type(absolute)
    def add_op(self, op, out, x):
        self._buffer_op("abs", x=x, out=out)

    @add_op.on_type(Add)
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

    @add_op.on_type(convolution1d)
    def add_op(self, op, output, input, filter):
        raise ValueError("Unhandled op: {}".format(op))

    @add_op.on_type(cos)
    def add_op(self, op, out, x):
        self._buffer_op("cos", x=x, out=out)

    @add_op.on_type(Dimshuffle)
    def add_op(self, op, out, x):
        self._buffer_op("dimshuffle", x=x, y=op.old_axis_positions, out=out)

    @add_op.on_type(Divide)
    def add_op(self, op, out, x, y):
        self._buffer_op("div", x=x, y=y, out=out)

    @add_op.on_type(Dot)
    def add_op(self, op, out, x, y):
        self._buffer_op("dot", x=x, y=y, out=out)

    @add_op.on_type(Equal)
    def add_op(self, op, out, x, y):
        self._buffer_op("eq", x=x, y=y, out=out)

    @add_op.on_type(exp)
    def add_op(self, op, out, x):
        self._buffer_op("exp", x=x, out=out)

    @add_op.on_type(Fill)
    def add_op(self, op, out, x):
        self._buffer_op("fill", x=op.scalar, out=x)

    @add_op.on_type(Greater)
    def add_op(self, op, out, x, y):
        self._buffer_op("gt", x=x, y=y, out=out)

    @add_op.on_type(GreaterEqual)
    def add_op(self, op, out, x, y):
        self._buffer_op("ge", x=x, y=y, out=out)

    @add_op.on_type(Less)
    def add_op(self, op, out, x, y):
        self._buffer_op("lt", x=x, y=y, out=out)

    @add_op.on_type(LessEqual)
    def add_op(self, op, out, x, y):
        self._buffer_op("le", x=x, y=y, out=out)

    @add_op.on_type(log)
    def add_op(self, op, out, x):
        self._buffer_op("log", x=x, out=out)

    @add_op.on_type(Max)
    def add_op(self, op, out, x):
        self._buffer_op("max", x=x, axis=0, out=out)

    @add_op.on_type(Maximum)
    def add_op(self, op, out, x, y):
        self._buffer_op("maximum", x=x, y=y, out=out)

    @add_op.on_type(Min)
    def add_op(self, op, out, x):
        self._buffer_op("min", x=x, axis=0, out=out)

    @add_op.on_type(Minimum)
    def add_op(self, op, out, x, y):
        self._buffer_op("minimum", x=x, y=y, out=out)

    @add_op.on_type(Multiply)
    def add_op(self, op, out, x, y):
        self._buffer_op("mul", x=x, y=y, out=out)

    @add_op.on_type(negative)
    def add_op(self, op, out, x):
        self._buffer_op("neg", x=x, out=out)

    @add_op.on_type(NotEqual)
    def add_op(self, op, out, x, y):
        self._buffer_op("ne", x=x, y=y, out=out)

    @add_op.on_type(onehot)
    def add_op(self, op, out, o, x):
        raise ValueError("Unhandled op: {}".format(op))

    @add_op.on_type(Power)
    def add_op(self, op, out, x, y):
        self._buffer_op("pow", x=x, y=y, out=out)

    @add_op.on_type(reciprocal)
    def add_op(self, op, out, x):
        self._buffer_op("rcp", x=x, out=out)

    @add_op.on_type(SetItem)
    def add_op(self, op, out, tensor, value):
        if op.item is None or op.item == _none_slice or op.item == ():
            self._buffer_op("assign", x=value, out=tensor)
        else:
            self._buffer_op("set_item", x=value, y=op.item, out=tensor)

    @add_op.on_type(sign)
    def add_op(self, op, out, x):
        self._buffer_op("sgn", x=x, out=out)

    @add_op.on_type(sin)
    def add_op(self, op, out, x):
        self._buffer_op("sin", x=x, out=out)

    @add_op.on_type(sqrt)
    def add_op(self, op, out, x):
        self._buffer_op("sqrt", x=x, out=out)

    @add_op.on_type(square)
    def add_op(self, op, out, x):
        self._buffer_op("sqr", x=x, out=out)

    @add_op.on_type(Subtract)
    def add_op(self, op, out, x, y):
        self._buffer_op("sub", x=x, y=y, out=out)

    @add_op.on_type(Sum)
    def add_op(self, op, out, x):
        self._buffer_op("sum", x=x, axis=0, out=out)

    @add_op.on_type(tanh)
    def add_op(self, op, out, x):
        self._buffer_op("tanh", x=x, out=out)

    @add_op.on_type(tensor_size)
    def add_op(self, op, out):
        self._buffer_op("fill", x=op.reduction_axes.size, out=out)

    @add_op.on_type(Unslice)
    def add_op(self, op, out, out_sliced, x):
        # out = self._cast_input(out)
        # out_sliced = self._cast_input(out_sliced)
        # x = self._cast_input(x)
        # self._buffer_op("fill", x=0, out=out)
        # self._buffer_op("set_item", x=x, out=out_sliced)
        raise ValueError("Unhandled op: {}".format(op))

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

    def bind_buffers(self):
        """
        Binds GPU addresses of buffers to the kernel parameters. When kernels
        and initial parameters are generated, tensors have not yet been
        allocated so a placeholder is used for the memory addresses. This must
        be called before the first kernel run to bind the tensor addresses in
        GPU memory to the kernel parameters.
        """
        if self.compound:
            for index in range(len(self.params)):
                if isinstance(self.params[index], TensorDescription):
                    self.params[index] = self.params[index].value.tensor.gpudata
        else:
            new_op = [self.ops_buffer[0][0]]
            for i in range(1, 5):
                if isinstance(self.ops_buffer[0][i], TensorDescription):
                    new_op.append(self.ops_buffer[0][i].value.tensor)
                else:
                    new_op.append(self.ops_buffer[0][i])
            self.ops_buffer[0] = tuple(new_op)
        self.buffers_bound = True

    def _buffer_op(self, op, x=None, y=None, out=None, axis=None):
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

        self.ops_buffer.append((op, x, y, out, axis))

    def compile(self):
        """
        Compiles ops buffer into a GPU kernel. First checks if this is a
        compound kernel which needs to be compiled. In cases where only a
        single special op are contained (dot, dimshuffle, etc) there is no
        compounding and the NervanaGPU implementation is called directly at run
        time.
        """
        if len(self.ops_buffer) == 0:
            return False

        if len(self.ops_buffer) == 1:
            if (self.ops_buffer[0][0] == "dot" or
                    self.ops_buffer[0][0] == "fill" or
                    self.ops_buffer[0][0] == "set_item" or
                    self.ops_buffer[0][0] == "dimshuffle"):
                self.compound = False

        if self.compound:
            # Remove duplicate tensor views
            tensors = dict()
            new_buffer = []
            for op in self.ops_buffer:
                new_op = list(op)
                for t in range(1, 4):
                    if isinstance(new_op[t], GPUTensor):
                        signature = (int(new_op[t].gpudata), new_op[t].shape, new_op[t].strides)
                        if signature in tensors.keys():
                            new_op[t] = tensors[signature]
                        else:
                            tensors[signature] = new_op[t]
                new_buffer.append(tuple(new_op))

            # Compile ops in compound buffer into a kernel
            self.kernel, self.params, self.shared_size = _prepare_compound_kernel(new_buffer)

        return True


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

    def __init__(self, transformer, kernels):
        self.transformer = transformer
        self.ng = transformer.ng
        self.kernels = kernels

    def __call__(self):
        for k in self.kernels:
            if not k.buffers_bound:
                k.bind_buffers()

            if k.compound:
                # Execute prepared kernel
                kernel = k.kernel
                params = k.params
                # import pdb; pdb.set_trace()
                kernel.prepared_async_call(*params, shared_size=k.shared_size)
            else:
                op = k.ops_buffer[0]
                if op[0] == "dot":
                    self.ng.compound_dot(op[1], op[2], op[3])
                elif op[0] == "fill":
                    op[3].fill(op[1])
                elif op[0] == "set_item":
                    op[3].__setitem__(op[2], op[1])
                elif op[0] == "dimshuffle":
                    if len(op[1].shape) == 2 and (op[1].shape[0] == 1 or op[1].shape[1] == 1):
                        if op[1].shape == op[3].shape:
                            op[3][:] = op[1]
                        else:
                            op[3][:] = op[1].T
                    else:
                        self.ng.copy_transpose(op[1], op[3], axes=op[2])


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
                self.__getitem__(key)._assign(value)

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

    @staticmethod
    def close_gpu():
        if GPUTransformer.__nervanagpu is not None:
            GPUTransformer.__nervanagpu.cleanup_backend()
            GPUTransformer.__nervanagpu = None

    def __init__(self, **kwargs):
        # TODO: Re-enable fusion
        # super(GPUTransformer, self).__init__(fusion=gpu_fusible, **kwargs)
        super(GPUTransformer, self).__init__(**kwargs)

        self.buffer_allocators = []
        self.kernel_groups = dict()
        self.tensors = dict()
        self.finished_transform = False
        self.current_buffer = None
        self.closed = False

        if GPUTransformer.__nervanagpu is None:
            GPUTransformer.__nervanagpu = NervanaGPU()
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
        kernels = []
        for fun in ordered_ops:
            if isinstance(fun, Function):
                # Iterate over compounded operations and build kernel for them
                kernel = GPUKernel(self)
                for op in fun.instructions:
                    out = op.tensor_description()
                    call_info = (_ for _ in op.call_info())
                    kernel.add_op(op, out, *call_info)
                if kernel.compile():
                    kernels.append(kernel)
            else:
                # Generate kernel for single operation
                out = fun.tensor_description()
                call_info = (_ for _ in fun.call_info())

                kernel = GPUKernel(self)
                kernel.add_op(fun, out, *call_info)
                if kernel.compile():
                    kernels.append(kernel)

        # Create kernel group
        kernel_group = GPUKernelGroup(self, kernels)
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
