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
from ngraph.transformers.gpu.util import _magic32, _flatten, _ceil_div

from neon.backends.kernels.cuda import pooling

from operator import itemgetter, mul
import numpy as np


class PoolFpropKernel(GPUKernel):
    """
    Kernel object to execute pooling forward propagation. Selects from Nervana's
    cuda pooling kernels.

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU
        op (PoolingOp): Graph op being transformed into this kernel

    Attributes:
        I (TensorDescriptionWrapper): Tensor for input feature maps
        O (TensorDescriptionWrapper): Tensor for output feature maps
        kernel (pycuda.driver.Function): Compiled GPU kernel to execute this
            pooling operation
        params (list): List of parameters to pass to kernel
    """
    def __init__(self, transformer, op):
        super(PoolFpropKernel, self).__init__(transformer)

        (self.I, ) = (_ for _ in op.call_info())
        self.O = op.tensor_description()
        self.dtype = self.O.dtype
        self.index = op.index

        if self.dtype.type is np.float16:
            clss = "hpool"
        elif self.dtype.type is np.float32:
            clss = "spool"
        else:
            raise TypeError("Type not supported {}".format(clss))

        C, D, H, W, _ = self.I.axes.lengths
        K, M, P, Q, N = self.O.axes.lengths

        J, T, R, S, pool_op = itemgetter(*('J', 'T', 'R', 'S', 'op'))(op.pool_params)
        pad_c, pad_d, pad_h, pad_w = \
            itemgetter(*('pad_' + s for s in ('c', 'd', 'h', 'w')))(op.pool_params)
        str_c, str_d, str_h, str_w = \
            itemgetter(*('str_' + s for s in ('c', 'd', 'h', 'w')))(op.pool_params)

        # default to non-overlapping
        if str_c is None:
            str_c = J
        if str_d is None:
            str_d = T
        if str_h is None:
            str_h = R
        if str_w is None:
            str_w = S

        self.overlap = 1.0

        # TODO: detect other forms of gaps
        if str_c > J or str_d > T or str_h > R or str_w > S:
            self.gaps = 1
        else:
            self.gaps = 0

        self.op   = pool_op
        self.C    = C
        self.K    = K
        self.M    = M
        self.P    = P
        self.Q    = Q
        self.JTRS = (J, T, R, S)
        self.DHW  = (D, H, W)
        self.MPQ  = (M, P, Q)
        self.padding = (pad_c, pad_d, pad_h, pad_w)
        self.strides = (str_c, str_d, str_h, str_w)

        self.dimI   = (C, D, H, W, N)
        self.dimO   = (K, M, P, Q, N)
        self.dimF2  = None
        self.dimI2  = (C * D * H * W, N)
        self.dimO2  = (K * M * P * Q, N)
        self.sizeI  = np.product(self.dimI)
        self.sizeO  = np.product(self.dimO)
        self.nOut   = np.product(self.MPQ) * K

        # precompute some multiplications for fast constant memory access
        WN   = W * N
        HWN  = H * WN
        DHWN = D * HWN
        RS   = R * S
        RST  = T * RS
        JRST = J * RST
        QN   = Q * N
        PQN  = P * QN
        MPQN = M * PQN

        assert JRST + 32 < 2**16, "Integer division is faster with 16bit numerators"

        sb_large = {
            #SB  shlP maskP shrP shlQ maskQ shrQ maskN shrN
            1  : (0,   0x00, 0,   0,   0x00, 0,   0xfff, 32), # 1x1  nnnnn
            2  : (0,   0x00, 0,   1,   0x10, 4,   0x00f,  4), # 1x2  xnnnn
            4  : (0,   0x00, 0,   2,   0x18, 3,   0x007,  3), # 1x4  xxnnn
            8  : (0,   0x00, 0,   3,   0x1c, 2,   0x003,  2), # 1x8  xxxnn
            16 : (0,   0x00, 0,   4,   0x1e, 1,   0x001,  1), # 1x16 xxxxn
            32 : (0,   0x00, 0,   5,   0x1f, 0,   0x000,  0), # 1x32 xxxxx
        }
        sb_medium = {
            #SB  shlP maskP shrP shlQ maskQ shrQ maskN shrN
            8  : (1,   0x10, 4,   2,   0x0c, 2,   0x003,  2), # 2x4  yxxnn
            16 : (1,   0x10, 4,   3,   0x0e, 1,   0x001,  1), # 2x8  yxxxn
            32 : (1,   0x10, 4,   4,   0x0f, 0,   0x000,  0), # 2x16 yxxxx
        }
        sb_small = {
            #SB  shlP maskP shrP shlQ maskQ shrQ maskN shrN
            16 : (2,   0x18, 3,   2,   0x06, 1,   0x001,  1), # 4x4  yyxxn
            32 : (2,   0x18, 3,   3,   0x07, 0,   0x000,  0), # 4x8  yyxxx
        }

        if N == 1:
            super_block = 0
        elif N < 32:
            super_block = len(bin(N - 1)) - 2
        else:
            super_block = 5
        super_block = 1 << (5 - super_block)

        # try to minimize the zero overlap in the superblock
        # but maximize the x dim of the superblock for more contiguous memory access
        if super_block < 8 or Q > 64:
            sb_params = sb_large.get(super_block)
        elif super_block < 16 or Q > 32:
            sb_params = sb_medium.get(super_block)
        else:
            sb_params = sb_small.get(super_block)

        supP = _ceil_div(P, 1 << sb_params[0])
        supQ = _ceil_div(Q, 1 << sb_params[3])

        # precompute the magic numbers and shift amounts for integer division
        magic_RST = _magic32(JRST + 32, RST)
        magic_RS  = _magic32(RST + 32, RS)
        magic_S   = _magic32(RS + 32, S)
        magic_P   = _magic32(M * supP, supP)

        fprop_name = "fprop_" + pool_op

        threads = 32 if super_block > 1 else N

        self.fprop_kernel = [fprop_name, (supQ, supP * M, K), (threads, 1, 1), _flatten([
            N, W, H, D, C, WN, HWN, DHWN,
            P, Q, magic_P, QN, PQN, MPQN,
            pad_c, pad_d, pad_h, pad_w,
            str_c, str_d, str_h, str_w,
            S, RS, RST, JRST, magic_S, magic_RS, magic_RST,
            supP, supQ, sb_params ])]

        lut_size = JRST
        if lut_size % 4 != 0:
            lut_size += 4 - lut_size % 4

        self.fprop_lut_size = super_block * lut_size * 4

        self.kernel = pooling.map_string2func(self.fprop_kernel[0],
                                              self.dtype.str[1:],
                                              self.transformer.ng.compute_capability)

    def bind_buffers(self):
        """
        Gets allocated tensors for input and output feature maps.
        Allocates a scratch tensor for argmax indices if the op is max pooling
        since this is required for bprop. Builds a final list of parameters to
        pass to the kernel.
        """
        I_data = self.I.value.tensor
        O_data = self.O.value.tensor

        # Allocate argmax tensor
        if self.op == "max":
            if self.index not in self.transformer.argmax_tensors:
                argmax = self.transformer.ng.empty_like(self.O.value.tensor)
                self.transformer.argmax_tensors[self.index] = argmax
            else:
                argmax = self.transformer.argmax_tensors[self.index]
            A_data = argmax.gpudata
        else:
            A_data = 0

        # TODO: argmax??
        kernel_args = self.fprop_kernel
        self.params = [kernel_args[1], kernel_args[2], None,
                       I_data.gpudata, O_data.gpudata, A_data, 1.0, 0.0, 0]
        self.params.extend(kernel_args[3])
        super(PoolFpropKernel, self).bind_buffers()

    def execute(self):
        """
        Executes the pooling kernel.
        """
        self.kernel.prepared_async_call(*self.params, shared_size=self.fprop_lut_size)


class PoolBpropKernel(GPUKernel):
    """
    Kernel object to execute pooling backward propagation. Selects from Nervana's
    cuda pooling kernels.

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU
        op (BpropPoolOp): Graph op being transformed into this kernel

    Attributes:
        I (TensorDescriptionWrapper): Tensor for input feature maps
        O (TensorDescriptionWrapper): Tensor for output feature maps
        kernel (pycuda.driver.Function): Compiled GPU kernel to execute this
            pooling operation
        params (list): List of parameters to pass to kernel
    """
    def __init__(self, transformer, op):
        super(PoolBpropKernel, self).__init__(transformer)

        (self.I, ) = (_ for _ in op.call_info())
        self.O = op.tensor_description()
        self.dtype = self.O.dtype
        self.index = op.index

        if self.dtype.type is np.float16:
            clss = "hpool"
        elif self.dtype.type is np.float32:
            clss = "spool"
        else:
            raise TypeError("Type not supported {}".format(clss))

        C, D, H, W, _ = self.O.axes.lengths
        K, M, P, Q, N = self.I.axes.lengths

        J, T, R, S, pool_op = itemgetter(*('J', 'T', 'R', 'S', 'op'))(op.pool_params)
        pad_c, pad_d, pad_h, pad_w = \
            itemgetter(*('pad_' + s for s in ('c', 'd', 'h', 'w')))(op.pool_params)
        str_c, str_d, str_h, str_w = \
            itemgetter(*('str_' + s for s in ('c', 'd', 'h', 'w')))(op.pool_params)

        # default to non-overlapping
        if str_c is None:
            str_c = J
        if str_d is None:
            str_d = T
        if str_h is None:
            str_h = R
        if str_w is None:
            str_w = S

        self.overlap = 1.0

        # TODO: detect other forms of gaps
        if str_c > J or str_d > T or str_h > R or str_w > S:
            self.gaps = 1
        else:
            self.gaps = 0

        self.op   = pool_op
        self.C    = C
        self.K    = K
        self.M    = M
        self.P    = P
        self.Q    = Q
        self.JTRS = (J, T, R, S)
        self.DHW  = (D, H, W)
        self.MPQ  = (M, P, Q)
        self.padding = (pad_c, pad_d, pad_h, pad_w)
        self.strides = (str_c, str_d, str_h, str_w)

        self.dimI   = (C, D, H, W, N)
        self.dimO   = (K, M, P, Q, N)
        self.dimF2  = None
        self.dimI2  = (C * D * H * W, N)
        self.dimO2  = (K * M * P * Q, N)
        self.sizeI  = np.product(self.dimI)
        self.sizeO  = np.product(self.dimO)
        self.nOut   = np.product(self.MPQ) * K

        # precompute some multiplications for fast constant memory access
        WN   = W * N
        HWN  = H * WN
        DHWN = D * HWN
        DH   = D * H
        RS   = R * S
        RST  = T * RS
        JRST = J * RST
        QN   = Q * N
        PQN  = P * QN
        MPQN = M * PQN

        assert JRST + 32 < 2**16, "Integer division is faster with 16bit numerators"

        sb_large = {
            #SB  shlP maskP shrP shlQ maskQ shrQ maskN shrN
            1  : (0,   0x00, 0,   0,   0x00, 0,   0xfff, 32), # 1x1  nnnnn
            2  : (0,   0x00, 0,   1,   0x10, 4,   0x00f,  4), # 1x2  xnnnn
            4  : (0,   0x00, 0,   2,   0x18, 3,   0x007,  3), # 1x4  xxnnn
            8  : (0,   0x00, 0,   3,   0x1c, 2,   0x003,  2), # 1x8  xxxnn
            16 : (0,   0x00, 0,   4,   0x1e, 1,   0x001,  1), # 1x16 xxxxn
            32 : (0,   0x00, 0,   5,   0x1f, 0,   0x000,  0), # 1x32 xxxxx
        }
        sb_medium = {
            #SB  shlP maskP shrP shlQ maskQ shrQ maskN shrN
            8  : (1,   0x10, 4,   2,   0x0c, 2,   0x003,  2), # 2x4  yxxnn
            16 : (1,   0x10, 4,   3,   0x0e, 1,   0x001,  1), # 2x8  yxxxn
            32 : (1,   0x10, 4,   4,   0x0f, 0,   0x000,  0), # 2x16 yxxxx
        }
        sb_small = {
            #SB  shlP maskP shrP shlQ maskQ shrQ maskN shrN
            16 : (2,   0x18, 3,   2,   0x06, 1,   0x001,  1), # 4x4  yyxxn
            32 : (2,   0x18, 3,   3,   0x07, 0,   0x000,  0), # 4x8  yyxxx
        }

        if N == 1:
            super_block = 0
        elif N < 32:
            super_block = len(bin(N - 1)) - 2
        else:
            super_block = 5
        super_block = 1 << (5 - super_block)

        # try to minimize the zero overlap in the superblock
        # but maximize the x dim of the superblock for more contiguous memory access
        if super_block < 8 or Q > 64:
            sb_params = sb_large.get(super_block)
        elif super_block < 16 or Q > 32:
            sb_params = sb_medium.get(super_block)
        else:
            sb_params = sb_small.get(super_block)

        supP = _ceil_div(P, 1 << sb_params[0])
        supQ = _ceil_div(Q, 1 << sb_params[3])

        # precompute the magic numbers and shift amounts for integer division
        magic_RST = _magic32(JRST + 32, RST)
        magic_RS  = _magic32(RST + 32, RS)
        magic_S   = _magic32(RS + 32, S)
        magic_P   = _magic32(M * supP, supP)

        bprop_name = "bprop_" + pool_op

        threads = 32 if super_block > 1 else N

        lut_size = JRST
        if lut_size % 4 != 0:
            lut_size += 4 - lut_size % 4

        self.bprop_lut_size = self.fprop_lut_size = super_block * lut_size * 4

        if self.overlap > 0:

            # we have a special kernel to handle the overlapping avg pooling
            bprop_name += "_overlap"

            magic_str_w = _magic32(W + S, str_w)
            magic_str_h = _magic32(H + R, str_h)
            magic_str_d = _magic32(D + T, str_d)
            magic_str_c = _magic32(C + J, str_c)

            if super_block > 1:

                bprop_name += "_smallN"

                if super_block < 8 or W > 64:
                    sb_params = sb_large.get(super_block)
                elif super_block < 16 or W > 32:
                    sb_params = sb_medium.get(super_block)
                else:
                    sb_params = sb_small.get(super_block)

                supH = _ceil_div(H, 1 << sb_params[0])
                supW = _ceil_div(W, 1 << sb_params[3])

                magic_H = _magic32(D * supH, supH)

                maxLutSize = \
                    _ceil_div(S, str_w) * \
                    _ceil_div(R, str_h) * \
                    _ceil_div(T, str_d) * \
                    _ceil_div(J, str_c)

                #neon_logger.display((supW, D*supH, C), sb_params, maxLutSize)

                self.bprop_kernel = [bprop_name, (supW, D * supH, C), (threads, 1, 1), _flatten([
                    N, W, H, D, C, WN, HWN, DHWN, magic_H,
                    pad_w, pad_h, pad_d, pad_c,
                    str_w, str_h, str_d, str_c,
                    magic_str_w, magic_str_h, magic_str_d, magic_str_c,
                    S, R, T, J, RS, RST, JRST, magic_S, magic_RS, magic_RST,
                    Q, P, M, K, QN, PQN, MPQN,
                    supH, supW, sb_params, maxLutSize])]

                lut_size = maxLutSize
                if lut_size % 4 != 0:
                    lut_size += 4 - lut_size % 4

                self.bprop_lut_size = super_block * lut_size * 4 * 2

            else:

                # The overlap kernel can be much more efficient if we aren't doing superblocking
                magic_H = _magic32(DH, H)

                self.bprop_kernel = [bprop_name, (W, DH, C), (threads, 1, 1), _flatten([
                    N, W, H, D, C, WN, HWN, DHWN, magic_H,
                    pad_w, pad_h, pad_d, pad_c,
                    str_w, str_h, str_d, str_c,
                    magic_str_w, magic_str_h, magic_str_d, magic_str_c,
                    S, R, T, J, RS, RST, JRST, magic_S, magic_RS, magic_RST,
                    Q, P, M, K, QN, PQN, MPQN])]

                self.bprop_lut_size = lut_size * 4 * 2
        else:
            self.bprop_kernel = [bprop_name, (supQ, supP * M, K), (threads, 1, 1), _flatten([
                N, W, H, D, C, WN, HWN, DHWN,
                P, Q, magic_P, QN, PQN, MPQN,
                pad_c, pad_d, pad_h, pad_w,
                str_c, str_d, str_h, str_w,
                S, RS, RST, JRST, magic_S, magic_RS, magic_RST,
                supP, supQ, sb_params])]

        self.kernel = pooling.map_string2func(self.bprop_kernel[0],
                                              self.dtype.str[1:],
                                              self.transformer.ng.compute_capability)

    def bind_buffers(self):
        """
        Gets allocated tensors for input and output feature maps.
        Allocates a scratch tensor for argmax indices if the op is max pooling
        since this is required for bprop. Builds a final list of parameters to
        pass to the kernel.
        """
        I_data = self.I.value.tensor
        O_data = self.O.value.tensor

        # Get argmax tensor
        if self.op == "max":
            assert self.index in self.transformer.argmax_tensors
            argmax = self.transformer.argmax_tensors[self.index]
            A_data = argmax.gpudata
        else:
            A_data = 0

        kernel_args = self.bprop_kernel
        self.params = [kernel_args[1], kernel_args[2], None,
                       I_data.gpudata, O_data.gpudata, A_data, 1.0, 0.0, 0]
        self.params.extend(kernel_args[3])
        super(PoolBpropKernel, self).bind_buffers()

    def execute(self):
        """
        Executes the pooling kernel.
        """
        self.kernel.prepared_async_call(*self.params, shared_size=self.bprop_lut_size)
