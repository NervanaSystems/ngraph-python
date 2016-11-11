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

from ngraph.transformers.gpu.kernel import GPUKernel
from ngraph.transformers.gpu.kernels import convolution

from operator import itemgetter
import numpy as np


class ConvFpropKernel(GPUKernel):
    """
    Kernel object to execute convolution forward propagation. Selects from Nervana's
    sass convolution/winograd kernels.

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU
        op (Convolution): Graph op being transformed into this kernel

    Attributes:
        I (TensorDescriptionWrapper): Tensor for input feature maps
        F (TensorDescriptionWrapper): Tensor for filters
        O (TensorDescriptionWrapper): Tensor for output feature maps
        fprop_kernels (KernelGroup): Kernel(s) used to execute fprop convolution
    """
    def __init__(self, transformer, op):
        super(ConvFpropKernel, self).__init__(transformer)

        self.O = op.tensor_description()
        self.I, self.F = (_ for _ in op.call_info())
        conv_dims = op.conv_params
        self.dtype = self.O.dtype

        C, D, H, W, N = self.I.axes.lengths
        C, T, R, S, K = self.F.axes.lengths
        K, M, P, Q, _ = self.O.axes.lengths
        pad_d, pad_h, pad_w = itemgetter(*('pad_' + s for s in ('d', 'h', 'w')))(conv_dims)
        str_d, str_h, str_w = itemgetter(*('str_' + s for s in ('d', 'h', 'w')))(conv_dims)

        args = (transformer.runtime, self.dtype, N, C, K, D, H, W, T, R, S,
                M, P, Q, pad_d, pad_h, pad_w, str_d, str_h, str_w)

        enable_winograd = transformer.runtime.enable_winograd
        use_cudac_kernels = transformer.runtime.use_cudac_kernels

        # ---- Cuda C ----
        if use_cudac_kernels:
            # 3D conv not supported yet
            if T > 1 or D > 1:
                raise ValueError("3D Convolution not supported by CUDA C kernels and "
                                 "pre-Maxwell GPUs")

            self.fprop_kernels = convolution.FpropCuda(*args)

        # ---- Winograd ----
        elif enable_winograd and R == 3 and S == 3 and \
                all(x == 1 for x in (D, M, T, str_w, str_h, str_d)):
            from .winograd_conv import (FpropWinograd_2x2_3x3, FpropWinograd_4x4_3x3)
            # Temp for now till we can autotune
            # 2 is safer for fp16 without batchnorm
            if self.dtype == np.float32 and enable_winograd == 4:
                winograd = 4
            else:
                winograd = 2

            if C < 8:
                self.fprop_kernels = convolution.FpropDirect(*args)
            elif winograd == 4 and H * W < 112 * 112:
                self.fprop_kernels = FpropWinograd_4x4_3x3(*args)
            else:
                self.fprop_kernels = FpropWinograd_2x2_3x3(*args)

        # ---- Direct ----
        else:
            self.fprop_kernels = convolution.FpropDirect(*args)

    def bind_buffers(self):
        """
        Gets GPUTensor for inputs and output. Binds all parameters to the GPU
        kernels
        """
        I_data = self.I.value.tensor
        F_data = self.F.value.tensor
        O_data = self.O.value.tensor
        self.fprop_kernels.bind_params(I_data, F_data, O_data, X=None,
                                       bias=None, bsum=None, alpha=1.0, beta=0.0,
                                       relu=False, brelu=False, slope=0.0)
        super(ConvFpropKernel, self).bind_buffers()

    def execute(self):
        """
        Call into convolution library to execute kernels
        """
        self.fprop_kernels.execute(1, unbind=False)


class ConvBpropKernel(GPUKernel):
    """
    Kernel object to execute convolution backward propagation. Selects from Nervana's
    sass convolution/winograd kernels.

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU
        op (bprop_conv): Graph op being transformed into this kernel

    Attributes:
        E (TensorDescriptionWrapper): Tensor for deltas from next layer
        F (TensorDescriptionWrapper): Tensor for filters
        O (TensorDescriptionWrapper): Tensor for output deltas
        bprop_kernels (KernelGroup): Kernel(s) used to execute bprop convolution
    """
    def __init__(self, transformer, op):
        super(ConvBpropKernel, self).__init__(transformer)

        self.O = op.tensor_description()
        self.E, self.F = (_ for _ in op.call_info())
        conv_dims = op.conv_params
        self.dtype = self.O.dtype

        C, D, H, W, N = self.O.axes.lengths
        C, T, R, S, K = self.F.axes.lengths
        K, M, P, Q, _ = self.E.axes.lengths
        pad_d, pad_h, pad_w = itemgetter(*('pad_' + s for s in ('d', 'h', 'w')))(conv_dims)
        str_d, str_h, str_w = itemgetter(*('str_' + s for s in ('d', 'h', 'w')))(conv_dims)

        args = (transformer.runtime, self.dtype, N, C, K, D, H, W, T, R, S,
                M, P, Q, pad_d, pad_h, pad_w, str_d, str_h, str_w)

        enable_winograd = transformer.runtime.enable_winograd
        use_cudac_kernels = transformer.runtime.use_cudac_kernels

        # ---- Cuda C ----
        if use_cudac_kernels:
            # 3D conv not supported yet
            if T > 1 or D > 1:
                raise ValueError("3D Convolution not supported by CUDA C kernels and "
                                 "pre-Maxwell GPUs")

            # TODO small C bprop?
            self.bprop_kernels = convolution.BpropCuda(*args)

        # ---- Winograd ----
        elif enable_winograd and R == 3 and S == 3 and \
                all(x == 1 for x in (D, M, T, str_w, str_h, str_d)):
            from .winograd_conv import (BpropWinograd_2x2_3x3, BpropWinograd_4x4_3x3)
            # Temp for now till we can autotune
            # 2 is safer for fp16 without batchnorm
            if self.dtype == np.float32 and enable_winograd == 4:
                winograd = 4
            else:
                winograd = 2

            if winograd == 4 and H * W < 112 * 112:
                self.bprop_kernels = BpropWinograd_4x4_3x3(*args)
            else:
                self.bprop_kernels = BpropWinograd_2x2_3x3(*args)

        # ---- Direct ----
        else:
            self.bprop_kernels = convolution.BpropDirect(*args)

    def bind_buffers(self):
        """
        Gets GPUTensor for inputs and output. Binds all parameters to the GPU
        kernels
        """
        E_data = self.E.value.tensor
        F_data = self.F.value.tensor
        O_data = self.O.value.tensor
        self.bprop_kernels.bind_params(E_data, F_data, O_data, O_data,
                                       bias=None, bsum=None, alpha=1.0, beta=0.0,
                                       relu=False, brelu=False, slope=0.0)
        super(ConvBpropKernel, self).bind_buffers()

    def execute(self):
        """
        Call into convolution library to execute kernels
        """
        self.bprop_kernels.execute(1, unbind=False)


class ConvUpdateKernel(GPUKernel):
    """
    Kernel object to execute convolution update to compute filter gradient. Selects
    from Nervana's sass convolution/winograd kernels.

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU
        op (update_conv): Graph op being transformed into this kernel

    Attributes:
        E (TensorDescriptionWrapper): Tensor for deltas from next layer
        I (TensorDescriptionWrapper): Tensor for input feature maps
        U (TensorDescriptionWrapper): Tensor for output gradient for update
        updat_kernels (KernelGroup): Kernel(s) used to execute convolution update
    """
    def __init__(self, transformer, op):
        super(ConvUpdateKernel, self).__init__(transformer)

        self.U = op.tensor_description()
        self.E, self.I = (_ for _ in op.call_info())
        conv_dims = op.conv_params
        self.dtype = self.U.dtype

        C, D, H, W, N = self.I.axes.lengths
        C, T, R, S, K = self.U.axes.lengths
        K, M, P, Q, _ = self.E.axes.lengths
        pad_d, pad_h, pad_w = itemgetter(*('pad_' + s for s in ('d', 'h', 'w')))(conv_dims)
        str_d, str_h, str_w = itemgetter(*('str_' + s for s in ('d', 'h', 'w')))(conv_dims)

        args = (transformer.runtime, self.dtype, N, C, K, D, H, W, T, R, S,
                M, P, Q, pad_d, pad_h, pad_w, str_d, str_h, str_w)

        enable_winograd = transformer.runtime.enable_winograd
        use_cudac_kernels = transformer.runtime.use_cudac_kernels

        # ---- Cuda C ----
        if use_cudac_kernels:
            # 3D conv not supported yet
            if T > 1 or D > 1:
                raise ValueError("3D Convolution not supported by CUDA C kernels "
                                 "and pre-Maxwell GPUs")

            self.updat_kernels = convolution.UpdateCuda(*args)

        # ---- Winograd ----
        elif enable_winograd and R == 3 and S == 3 and \
                all(x == 1 for x in (D, M, T, str_w, str_h, str_d)):
            from .winograd_conv import (UpdateWinograd_3x3_2x2, UpdateWinograd_3x3_4x4)
            # Temp for now till we can autotune
            # 2 is safer for fp16 without batchnorm
            if self.dtype == np.float32 and enable_winograd == 4:
                winograd = 4
            else:
                winograd = 2

            if N >= 4 and (C < 8 or H * W > 112 * 112):
                self.updat_kernels = convolution.UpdateDirect(*args)
            elif winograd == 4:
                self.updat_kernels = UpdateWinograd_3x3_4x4(*args)
            else:
                self.updat_kernels = UpdateWinograd_3x3_2x2(*args)

        # ---- Direct ----
        else:
            if N >= 4:
                self.updat_kernels = convolution.UpdateDirect(*args)
            else:
                raise NotImplementedError("This is not supported")

    def bind_buffers(self):
        """
        Gets GPUTensor for inputs and output. Binds all parameters to the GPU
        kernels
        """
        E_data = self.E.value.tensor
        I_data = self.I.value.tensor
        U_data = self.U.value.tensor
        self.updat_kernels.bind_params(I_data, E_data, U_data, alpha=1.0, beta=0.0)
        super(ConvUpdateKernel, self).bind_buffers()

    def execute(self):
        """
        Call into convolution library to execute kernels
        """
        self.updat_kernels.execute(1, unbind=False)
