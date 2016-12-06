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

from ngraph.transformers.gpu.kernel import GPUKernel, pointer_from_td
from ngraph.transformers.gpu.float_ew2 import TensorDescriptionWrapper
from ngraph.transformers.gpu.kernels.cuda.copy_transpose import _get_copy_transpose_kernel
from ngraph.op_graph.axes import TensorDescription

import numpy as np


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

        out = TensorDescriptionWrapper(op.tensor_description(), 2)
        (arg, ) = (_ for _ in op.call_info())
        in_tensor = TensorDescriptionWrapper(arg, 2)

        dtype = out.dtype
        shape = in_tensor.shape
        axes = op.old_axis_positions

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


class FillKernel(GPUKernel):
    """
    Kernel used to fill a tensor with a scalar value.

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU
        td (TensorDescription): Tensor to fill
        value : Scalar value used to fill tensor

    Attributes:
        value : Scalar value to fill tensor
        out (GPUTensor): Tensor to fill with value
    """
    def __init__(self, transformer, td, value):
        super(FillKernel, self).__init__(transformer)

        self.value = value
        self.out = td

    def bind_buffers(self):
        """
        Get allocated GPU tensor for output
        """
        self.out = self.out.value.tensor
        super(FillKernel, self).bind_buffers()

    def execute(self):
        """
        Use memset driver functions to fill tensor with scalar
        Temporarily uses neon GPUTensor's fill method
        """
        self.out.fill(self.value)


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


class UnsliceKernel(GPUKernel):
    """
    Kernel used for Unslice op

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU
        op (Unslice): Graph op being transformed into this kernel

    Attributes:
        out (GPUTensor): Full dest tensor
        out_sliced (GPUTensor): Slice of dest tensor which is the same shape as
            the source tensor
        value (GPUTensor: Source tensor
    """
    def __init__(self, transformer, op):
        super(UnsliceKernel, self).__init__(transformer)

        self.out_sliced, self.x = (_ for _ in op.call_info())
        self.out = op.tensor_description()

        # Use copy transpose kernel for unsupported cases
        if len(self.out_sliced.strides) > 0 and np.min(self.out_sliced.strides) < 0:
            dtype = self.out_sliced.dtype
            shape = self.out_sliced.shape
            axes = range(len(self.out_sliced.shape))

            self.kernel, self.params = get_dimshuffle(dtype, shape, tuple(axes),
                                                      TensorDescriptionWrapper(self.x),
                                                      TensorDescriptionWrapper(self.out_sliced))
        else:
            self.kernel = None

    def bind_buffers(self):
        """
        Get allocated GPU tensor for source and dest
        """
        if isinstance(self.out_sliced, TensorDescription):
            self.out_sliced = self.out_sliced.value
        if isinstance(self.x, TensorDescription):
            self.x = self.x.value.tensor
        if isinstance(self.out, TensorDescription):
            self.out = self.out.value.tensor

        if self.kernel is not None:
            for index in range(len(self.params)):
                if isinstance(self.params[index], TensorDescription):
                    self.params[index] = pointer_from_td(self.params[index])

        super(UnsliceKernel, self).bind_buffers()

    def execute(self):
        """
        Run fill and SetItem kernel to execute the Unslice operation
        """
        self.out.fill(0)
        if self.kernel is not None:
            self.kernel.prepared_async_call(self.kernel.grid, self.kernel.block,
                                            None, *self.params)
        elif self.out_sliced.shape == ():
            self.out_sliced.tensor.fill(self.x.get())
        else:
            self.out_sliced[:] = self.x
