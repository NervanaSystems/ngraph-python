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

from ngraph.transformers.gpu.conv import ConvFpropKernel, ConvBpropKernel, ConvUpdateKernel
from ngraph.transformers.gpu.kernels import kernel_specs
from ngraph.transformers.gpu.float_ew2 import TensorDescriptionWrapper, _get_register_type, \
    FlexPtrDescription
from ngraph.transformers.gpu.kernels.convolution import _magic32, _magic64, _get_sm_count

from pycuda.compiler import SourceModule
from pycuda.tools import context_dependent_memoize
import pycuda.driver as drv

from operator import mul
import numpy as np
import sys

if sys.version_info >= (3, 0):
    from functools import reduce


class ScratchBufferWrapper(object):
    def __init__(self, size, offset, runtime):
        self.size = size
        self.offset = offset
        self.runtime = runtime

        runtime.scratch_buffer_reset()
        runtime.set_scratch_size(size + offset)

    def get_ptr(self):
        self.runtime.scratch_buffer_init()
        return self.runtime.scratch_buffer(self.size + self.offset) + self.offset


class FlexConvFpropKernel(ConvFpropKernel):
    """
    Inherits all parent class ConvFpropKernel behavior except
    selects flex convolution kernel and sets up flex parameters
    """
    def __init__(self, transformer, op):
        self.alpha = 1.0
        self.beta = 0.0
        # flex does not support dilated convolution
        msg = 'flexsim does not support dilated convolution'
        assert op.conv_params['dil_d'] == 1, msg
        assert op.conv_params['dil_h'] == 1, msg
        assert op.conv_params['dil_w'] == 1, msg
        super(FlexConvFpropKernel, self).__init__(transformer, op)

    def gen_kernels(self, runtime, N, C, K, D, H, W, T, R, S, M, P, Q,
                    pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w):
        self.I = TensorDescriptionWrapper(self.I, len(self.I.shape))
        self.F = TensorDescriptionWrapper(self.F, len(self.F.shape))
        self.O = TensorDescriptionWrapper(self.O, len(self.O.shape))

        self.flex_entry_I = self.I.flex_entry()
        self.flex_entry_F = self.F.flex_entry()
        self.flex_entry_O = self.O.flex_entry()

        vec_size = 4 if self.dtype.itemsize == 4 else 8

        assert N % 32 == 0, "N dim must be multiple of 32"
        assert K % vec_size == 0, "K dim must be multiple of %d" % vec_size

        if self.dtype.type == "flex":
            clss = "fconv"
        else:
            raise TypeError("Type not supported.")

        self.C = C
        self.K = K
        self.M = M
        self.P = P
        self.Q = Q
        self.NCK = (N, C, K)
        self.TRS = (T, R, S)
        self.DHW = (D, H, W)
        self.MPQ = (M, P, Q)
        self.padding = (pad_d, pad_h, pad_w)
        self.strides = (str_d, str_h, str_w)

        self.all_params = (N, C, K, D, H, W, T, R, S, pad_d, pad_h, pad_w, str_d, str_h, str_w)

        self.dimI = (C, D, H, W, N)
        self.dimF = (C, T, R, S, K)
        self.dimF = (K, T, R, S, C)
        self.dimO = (K, M, P, Q, N)
        self.dimI2 = (C * D * H * W, N)
        self.dimF2 = (C * T * R * S, K)
        self.dimF2t = (K, C * T * R * S)
        self.dimO2 = (K * M * P * Q, N)
        self.dimS = (K, 1)
        self.sizeI = reduce(mul, self.dimI, 1)
        self.sizeF = reduce(mul, self.dimF, 1)
        self.sizeO = reduce(mul, self.dimO, 1)
        self.nOut = reduce(mul, self.MPQ, 1) * K

        # precompute some multiplications for fast constant memory access
        WN = W * N
        HWN = H * WN
        DHWN = D * HWN
        RS = R * S
        RST = T * RS
        CRST = C * RST
        KRST = K * RST
        PQ = P * Q
        PQM = M * PQ
        QN = Q * N
        PQN = P * QN
        MPQN = M * PQN

        if CRST > 2**16:
            assert CRST < 2**16, "Integer division is faster with 16bit numerators"

        # precompute the magic numbers and shift amounts for integer division
        magic_PQ = _magic64(PQ)
        magic_Q = _magic64(Q)
        magic_RS = _magic32(RST + 32, RS)
        magic_S = _magic32(RS + 32, S)

        # flop count for benchmarking
        self.flops = PQM * K * N * CRST * 2.0

        tile_N = 128 if N > 64 else 64
        grid_N = _grid_dim(tile_N, N)
        tiles_CK = (128, 64, 32) if tile_N == 128 else (128, 64)

        # FPROP #
        self.fprop_kernels = kernel_specs.xprop_conv_kernels(
            clss, "fprop", "K", tile_N, grid_N, K, tiles_CK, PQM, RST,
            _flatten([N, K, D, H, W, WN, HWN, DHWN,
                      C, KRST, RST, RS, magic_RS, S, magic_S,
                      pad_d, pad_h, pad_w, str_d, str_h, str_w,
                      Q, PQ, QN, PQN, MPQN, magic_Q, magic_PQ]))

        # shared lookup table size
        self.fprop_lut_size = RST * 4 * 2

        # Set to 5 for the current T1000 HW config
        self.trunc_rows = 32
        flags = self.trunc_rows << 8

        self.kernels = []
        for kernel in self.fprop_kernels:
            # TODO: Populate alpha and beta parameters (in a separate loop!).
            # alpha (used to be params[6]) will be multiplied with
            self.kernels.append([
                kernel_specs.get_kernel(kernel[0]), kernel[1], kernel[2], None,
                0, self.O, self.I, self.F, 1.0, 0.0, flags,
                kernel[3]] + kernel[4])

        for kernel in self.kernels:
            kernel.extend((FlexPtrDescription(self.flex_entry_O), 1.0))
            kernel[10] &= 0xfffffffe  # Enable output flag

        # record output flex id for autoflex
        self.output_flex_ids = [self.flex_entry_O.flex_id]

    def bind_buffers(self):
        for k_id in range(len(self.kernels)):
            for index in range(len(self.kernels[k_id])):
                if isinstance(self.kernels[k_id][index], TensorDescriptionWrapper):
                    self.kernels[k_id][index] = self.kernels[k_id][index].td.value.tensor.gpudata

                if isinstance(self.kernels[k_id][index], ScratchBufferWrapper):
                    self.kernels[k_id][index] = self.kernels[k_id][index].get_ptr()

        self.buffers_bound = True

    def bind_flex_scales(self):
        scaleAB = self.flex_entry_I.scale * self.flex_entry_F.scale
        scaleC = self.flex_entry_O.scale
        alpha = self.alpha * scaleAB
        beta = self.beta * scaleC

        for kernel in self.kernels:
            kernel[8] = alpha
            kernel[9] = beta
            kernel[-1] = 1. / scaleC

        for kernel in self.kernels:
            FlexPtrDescription.bind_ptr(kernel)

    def execute(self):
        for kernel in self.kernels:
            kernel[0].prepared_async_call(*kernel[1:], shared_size=self.fprop_lut_size)


class FlexConvBpropKernel(ConvBpropKernel):
    """
    Inherits all parent class ConvBpropKernel behavior except
    selects flex convolution kernel and sets up flex parameters
    """
    def __init__(self, transformer, op):
        self.alpha = 1.0
        self.beta = 0.0
        super(FlexConvBpropKernel, self).__init__(transformer, op)

    def gen_kernels(self, runtime, N, C, K, D, H, W, T, R, S, M, P, Q,
                    pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w):
        self.E = TensorDescriptionWrapper(self.E, len(self.E.shape))
        self.F = TensorDescriptionWrapper(self.F, len(self.F.shape))
        self.O = TensorDescriptionWrapper(self.O, len(self.O.shape))

        self.flex_entry_E = self.E.flex_entry()
        self.flex_entry_F = self.F.flex_entry()
        self.flex_entry_O = self.O.flex_entry()

        F_size = int(np.prod(self.F.shape) * 2)
        O_size = int(np.prod(self.O.shape) * 2)

        vec_size = 4 if self.dtype.itemsize == 4 else 8

        assert N % 32 == 0, "N dim must be multiple of 32"
        assert K % vec_size == 0, "K dim must be multiple of %d" % vec_size

        if self.dtype.type == "flex":
            clss = "fconv"
        else:
            raise TypeError("Type not supported.")

        self.C = C
        self.K = K
        self.M = M
        self.P = P
        self.Q = Q
        self.NCK = (N, C, K)
        self.TRS = (T, R, S)
        self.DHW = (D, H, W)
        self.MPQ = (M, P, Q)
        self.padding = (pad_d, pad_h, pad_w)
        self.strides = (str_d, str_h, str_w)

        self.all_params = (N, C, K, D, H, W, T, R, S, pad_d, pad_h, pad_w, str_d, str_h, str_w)

        self.dimI = (C, D, H, W, N)
        self.dimF = (C, T, R, S, K)
        self.dimFb = (K, T, R, S, C)
        self.dimO = (K, M, P, Q, N)
        self.dimI2 = (C * D * H * W, N)
        self.dimF2 = (C * T * R * S, K)
        self.dimF2t = (K, C * T * R * S)
        self.dimO2 = (K * M * P * Q, N)
        self.dimS = (K, 1)
        self.sizeI = reduce(mul, self.dimI, 1)
        self.sizeF = reduce(mul, self.dimF, 1)
        self.sizeO = reduce(mul, self.dimO, 1)
        self.nOut = reduce(mul, self.MPQ, 1) * K

        # precompute some multiplications for fast constant memory access
        HW = H * W
        DHW = D * HW
        WN = W * N
        HWN = H * WN
        DHWN = D * HWN
        RS = R * S
        RST = T * RS
        CRST = C * RST
        PQ = P * Q
        PQM = M * PQ
        QN = Q * N
        PQN = P * QN
        MPQN = M * PQN

        if CRST > 2**16:
            assert CRST < 2**16, "Integer division is faster with 16bit numerators"

        # precompute the magic numbers and shift amounts for integer division
        magic_HW = _magic64(HW)
        magic_W = _magic64(W)
        magic_PQ = _magic64(PQ)
        magic_Q = _magic64(Q)
        magic_RST = _magic32(CRST, RST)
        magic_RS = _magic32(RST + 32, RS)
        magic_S = _magic32(RS + 32, S)
        magic_str_w = _magic32(W + S, str_w)
        magic_str_h = _magic32(H + R, str_h)
        magic_str_d = _magic32(D + T, str_d)

        # flop count for benchmarking
        self.flops = PQM * K * N * CRST * 2.0

        tile_N = 128 if N > 64 else 64
        grid_N = _grid_dim(tile_N, N)
        tiles_CK = (128, 64, 32) if tile_N == 128 else (128, 64)

        # BPROP #
        if C < 16 or C % vec_size != 0:
            # special kernel for deconv into first layer
            kernel_name = "%s_bprop_C1_N64" % clss

            grid = (PQM, _grid_dim(32, CRST), _grid_dim(64, N))
            block = (32, 1, 1)

            self.bprop_kernels = [[kernel_name, grid, block, 0, _flatten([
                N, K, D, H, W, WN, HWN, DHWN,
                C, CRST, RST, magic_RST, RS, magic_RS, S, magic_S,
                pad_d, pad_h, pad_w, str_d, str_h, str_w,
                Q, PQ, QN, PQN, MPQN, magic_Q, magic_PQ,
                CRST * 8 * self.dtype.itemsize, MPQN * 8 * self.dtype.itemsize])]]

            # generate the kernel args for transpose CRST,K => K,CRST
            self.shuffle_args = [CRST, K]
            gridX = (K >> 5) + (K & 31 != 0)
            gridY = (CRST >> 5) + (CRST & 31 != 0)
            self.shuffle_grid = (gridX, gridY, 1)
            self.shuffle_block = (32, 8, 1)
            self.bprop_zero = self.sizeI * self.dtype.itemsize
            self.bprop_lut_size = 0

        else:
            self.bprop_kernels = kernel_specs.xprop_conv_kernels(
                clss, "bprop", "C", tile_N, grid_N, C, tiles_CK, DHW, RST, _flatten([
                    N, C, M, P, Q, QN, PQN, MPQN,
                    K, CRST, RST, RS, magic_RS, S, magic_S,
                    pad_d, pad_h, pad_w, str_d, str_h, str_w,
                    W, HW, WN, HWN, DHWN, magic_W, magic_HW,
                    R, T, magic_str_w, magic_str_h, magic_str_d]))

            # generate the kernel args for dim shuffling CRSTK => KRSTC
            self.shuffle_args = _flatten([
                RST * K, RS * K, S * K, K,
                RST * C, RS * C, S * C, C,
                RS, magic_RS, S, magic_S])
            gridX = (K >> 5) + (K & 31 != 0)
            gridY = (C >> 5) + (C & 31 != 0)
            self.shuffle_grid = (gridX, gridY, RST)
            self.shuffle_block = (32, 8, 1)
            self.bprop_zero = 0
            self.bprop_lut_size = RST * 4 * 2

        # Set to 5 for the current T1000 HW config
        self.trunc_rows = 32
        flags = self.trunc_rows << 8

        # Must dim shuffle filter data for bprop kernel
        F_data = ScratchBufferWrapper(F_size, 0, runtime)
        if self.bprop_zero:
            Out = ScratchBufferWrapper(O_size, F_size, runtime)
            shuffle_kernel = _get_transpose_kernel(self.dtype)
        else:
            Out = self.O
            # can point to transpose or dimshuffle kernel
            shuffle_kernel = _get_shuffle_kernel(self.dtype)
        shuffle_args = [self.shuffle_grid, self.shuffle_block, None,
                        F_data, self.F] + self.shuffle_args
        shuffle_kernel = [shuffle_kernel] + shuffle_args

        # Have to zero output buffer and use type conversion for kernel using atomics
        if self.bprop_zero:
            shape = [int(np.prod(self.O.shape[:-1])), self.O.shape[-1]]
            convert_kernel = _prepare_convert_kernel(Out, "f2", self.O, shape,
                                                     FlexPtrDescription(self.flex_entry_O))
            self.convert_out = True
        else:
            self.convert_out = False

        self.kernels = []
        for kernel in self.bprop_kernels:
            # TODO: Populate alpha and beta parameters (in a separate loop!).
            # alpha (used to be params[6]) will be multiplied with
            self.kernels.append([
                kernel_specs.get_kernel(kernel[0]), kernel[1], kernel[2], None,
                0, Out, self.E, F_data, 1.0, 0.0, flags, kernel[3]] + kernel[4])

        for kernel in self.kernels:
            kernel.extend((FlexPtrDescription(self.flex_entry_O), 1.0))
            kernel[10] &= 0xfffffffe  # Enable output flag

        self.kernels = [shuffle_kernel] + self.kernels
        if self.convert_out:
            self.kernels.append(convert_kernel)

        # record output flex id for autoflex
        self.output_flex_ids = [self.flex_entry_O.flex_id]

    def bind_buffers(self):
        for k_id in range(len(self.kernels)):
            for index in range(len(self.kernels[k_id])):
                if isinstance(self.kernels[k_id][index], TensorDescriptionWrapper):
                    self.kernels[k_id][index] = self.kernels[k_id][index].td.value.tensor.gpudata

                if isinstance(self.kernels[k_id][index], ScratchBufferWrapper):
                    self.kernels[k_id][index] = self.kernels[k_id][index].get_ptr()

        self.buffers_bound = True

    def bind_flex_scales(self):
        scaleAB = self.flex_entry_E.scale * self.flex_entry_F.scale
        scaleC = self.flex_entry_O.scale
        alpha = self.alpha * scaleAB
        beta = self.beta * scaleC

        for kernel in self.kernels[1:1 + len(self.bprop_kernels)]:
            kernel[8] = alpha
            kernel[9] = beta
            kernel[-1] = 1. / scaleC

        if self.convert_out:
            self.kernels[-1][-2] = 1. / scaleC

        for kernel in self.kernels:
            FlexPtrDescription.bind_ptr(kernel)

    def execute(self):
        if self.bprop_zero:
            self.O.td.value[:] = 0

        for kernel in self.kernels:
            kernel[0].prepared_async_call(*kernel[1:], shared_size=self.bprop_lut_size)


class FlexConvUpdateKernel(ConvUpdateKernel):
    """
    Inherits all parent class ConvUpdateKernel behavior except
    selects flex convolution kernel and sets up flex parameters
    """
    def __init__(self, transformer, op):
        self.alpha = 1.0
        self.beta = 0.0
        super(FlexConvUpdateKernel, self).__init__(transformer, op)

    def gen_kernels(self, runtime, N, C, K, D, H, W, T, R, S, M, P, Q,
                    pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w):
        self.I = TensorDescriptionWrapper(self.I, len(self.I.shape))
        self.E = TensorDescriptionWrapper(self.E, len(self.E.shape))
        self.U = TensorDescriptionWrapper(self.U, len(self.U.shape))

        self.flex_entry_I = self.I.flex_entry()
        self.flex_entry_E = self.E.flex_entry()
        self.flex_entry_U = self.U.flex_entry()

        U_size = int(np.prod(self.U.shape) * 4)

        vec_size = 4 if self.dtype.itemsize == 4 else 8

        assert N % 32 == 0, "N dim must be multiple of 32"
        assert K % vec_size == 0, "K dim must be multiple of %d" % vec_size

        if self.dtype.type == "flex":
            clss = "fconv"
        else:
            raise TypeError("Type not supported.")

        self.C = C
        self.K = K
        self.M = M
        self.P = P
        self.Q = Q
        self.NCK = (N, C, K)
        self.TRS = (T, R, S)
        self.DHW = (D, H, W)
        self.MPQ = (M, P, Q)
        self.padding = (pad_d, pad_h, pad_w)
        self.strides = (str_d, str_h, str_w)

        self.all_params = (N, C, K, D, H, W, T, R, S, pad_d, pad_h, pad_w, str_d, str_h, str_w)

        self.dimI = (C, D, H, W, N)
        self.dimF = (C, T, R, S, K)
        self.dimFb = (K, T, R, S, C)
        self.dimO = (K, M, P, Q, N)
        self.dimI2 = (C * D * H * W, N)
        self.dimF2 = (C * T * R * S, K)
        self.dimF2t = (K, C * T * R * S)
        self.dimO2 = (K * M * P * Q, N)
        self.dimS = (K, 1)
        self.sizeI = reduce(mul, self.dimI, 1)
        self.sizeF = reduce(mul, self.dimF, 1)
        self.sizeO = reduce(mul, self.dimO, 1)
        self.nOut = reduce(mul, self.MPQ, 1) * K

        # precompute some multiplications for fast constant memory access
        WN = W * N
        HWN = H * WN
        DHWN = D * HWN
        RS = R * S
        RST = T * RS
        CRST = C * RST
        CRSTK = K * CRST
        PQ = P * Q
        PQM = M * PQ
        QN = Q * N
        PQN = P * QN
        MPQN = M * PQN

        if CRST > 2**16:
            assert CRST < 2**16, "Integer division is faster with 16bit numerators"

        # precompute the magic numbers and shift amounts for integer division
        magic_RST = _magic32(CRST, RST)
        magic_RS = _magic32(RST + 32, RS)
        magic_S = _magic32(RS + 32, S)

        # flop count for benchmarking
        self.flops = PQM * K * N * CRST * 2.0

        # UPDATE #

        grid_C = _grid_dim(128, CRST)
        sm_count = _get_sm_count()

        # in float32 for big feature_map layers the smaller tile is actually faster
        # so restrict tile selection to just that.
        if self.dtype.type is np.float32 and PQ > 56 * 56:
            K_tiles = (64,)
        else:
            K_tiles = (128, 64)

        determ = ""
        self.determ = 0

        self.updat_kernels = []
        for tile_K, grid_K, offset_K in kernel_specs.K_partitions(K, K_tiles):

            kernel_name = "%s_updat%s_C128_K%d" % (clss, determ, tile_K)
            base_blocks = M * grid_C * grid_K

            grid_P, grid_Q, threads = kernel_specs.update_grid(kernel_name,
                                                               base_blocks, P, Q, sm_count)

            grid_PQ = grid_P * grid_Q
            magic_PQu = _magic64(grid_PQ)
            magic_Qu = _magic64(grid_Q)

            block = (threads, 1, 1)
            if RST > 1:
                grid = (M * grid_PQ, grid_C, grid_K)
            else:
                grid = (grid_C, grid_K, M * grid_PQ)

            self.determ *= M * grid_PQ
            self.determ_shape = (M * grid_PQ, CRSTK)

            self.updat_kernels.append([kernel_name, grid, block, offset_K, _flatten([
                N, K, D, H, W, WN, HWN, DHWN,
                C, CRST, RST, magic_RST, RS, magic_RS, S, magic_S,
                pad_d, pad_h, pad_w, str_d, str_h, str_w,
                P, Q, PQ, QN, PQN, MPQN, magic_Qu, magic_PQu,
                grid_P, grid_Q, grid_PQ])])

        # Set to 5 for the current T1000 HW config
        self.trunc_rows = 32
        flags = self.trunc_rows << 8

        # Have to convert output from float to flex
        U_data = ScratchBufferWrapper(U_size, 0, runtime)
        shape = [int(np.prod(self.U.shape[:-1])), self.U.shape[-1]]
        convert_kernel = _prepare_convert_kernel(U_data, "f4", self.U, shape,
                                                 FlexPtrDescription(self.flex_entry_U))

        self.kernels = []
        for kernel in self.updat_kernels:
            # TODO: Populate alpha and beta parameters (in a separate loop!).
            # alpha (used to be params[6]) will be multiplied with
            self.kernels.append([
                kernel_specs.get_kernel(kernel[0]), kernel[1], kernel[2], None,
                0, U_data, self.I, self.E, 1.0, 0.0, flags,
                kernel[3]] + kernel[4])

        for kernel in self.kernels:
            kernel.extend((FlexPtrDescription(self.flex_entry_U), 1.0))
            kernel[10] &= 0xfffffffe  # Enable output flag

        self.kernels.append(convert_kernel)

        # record output flex id for autoflex
        self.output_flex_ids = [self.flex_entry_U.flex_id]

    def bind_buffers(self):
        for k_id in range(len(self.kernels)):
            for index in range(len(self.kernels[k_id])):
                if isinstance(self.kernels[k_id][index], TensorDescriptionWrapper):
                    self.kernels[k_id][index] = self.kernels[k_id][index].td.value.tensor.gpudata

                if isinstance(self.kernels[k_id][index], ScratchBufferWrapper):
                    self.kernels[k_id][index] = self.kernels[k_id][index].get_ptr()

        self.buffers_bound = True

    def bind_flex_scales(self):
        scaleAB = self.flex_entry_I.scale * self.flex_entry_E.scale
        scaleC = self.flex_entry_U.scale
        alpha = self.alpha * scaleAB
        beta = self.beta * scaleC

        for kernel in self.kernels[:-1]:
            kernel[8] = alpha
            kernel[9] = beta
            kernel[-1] = 1. / scaleC

        self.kernels[-1][-2] = 1. / scaleC

        for kernel in self.kernels:
            FlexPtrDescription.bind_ptr(kernel)

    def execute(self):
        # This zeros out the scratch buffer which is accumulated into using atomics
        # for update output kernels
        drv.memset_d32(self.kernels[0][5], 0, int(np.prod(self.U.shape)))

        for kernel in self.kernels:
            kernel[0].prepared_async_call(*kernel[1:])


def _grid_dim(tile_size, dim_size):
    return dim_size // tile_size + (dim_size % tile_size != 0)


def _flatten(lst):
    """
    flatten a nested list of lists or values
    """
    return sum(([x] if not isinstance(x, (list, tuple))
                else _flatten(x) for x in lst), [])


_transpose_kernel = r"""
__global__ void transpose(%(type)s* out, const %(type)s* in, int rows, int cols)
{
    __shared__ %(type)s tile[32][33];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int gx = bx * 32 + tx;
    int gy = by * 32 + ty;

    for (int j = 0; j < 32; j += 8)
    {
        int gy8 = gy + j;
        if (gy8 < rows && gx < cols)
            tile[ty + j][tx] = in[gy8*cols + gx];
    }
    __syncthreads();

    gx = by * 32 + tx;
    gy = bx * 32 + ty;

    for (int j = 0; j < 32; j += 8)
    {
        int gy8 = gy + j;
        if (gy8 < cols && gx < rows)
            out[gy8*rows + gx] = tile[tx][ty + j];
    }
}
"""


@context_dependent_memoize
def _get_transpose_kernel(dtype):

    code = _transpose_kernel % {
        "type": _get_register_type(dtype, memory=True)
    }
    module = SourceModule(code)
    kernel = module.get_function("transpose")
    kernel.prepare("PPII")
    return kernel


_shuffle_kernel = r"""
__global__ void dimShuffle(
    %(type)s* out, const %(type)s* in,
    int TRSK, int RSK, int SK, int K,
    int TRSC, int RSC, int SC, int C,
    int RS, int magic_RS, int shift_RS,
    int S,  int magic_S,  int shift_S)
{
    __shared__ %(type)s tile[32][33];

    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int bk  = blockIdx.x;
    int bc  = blockIdx.y;
    int trs = blockIdx.z;

    int k  = bk * 32 + tx;
    int c  = bc * 32 + ty;

    int t  = magic_RS * trs; t >>= shift_RS;
    int rs = trs - t*RS;

    int r = magic_S * rs; r >>= shift_S;
    int s = rs - r*S;

    for (int j = 0; j < 32; j += 8)
    {
        int cj = c + j;
        if (cj < C && k < K)
            tile[ty + j][tx] = in[ cj*TRSK + t*RSK + r*SK + s*K + k ];
    }
    __syncthreads();

    k = bk * 32 + ty;
    c = bc * 32 + tx;

    for (int i = 0; i < 32; i += 8)
    {
        int ki = k + i;
        if (ki < K && c < C)
            out[ ki*TRSC + t*RSC + r*SC + s*C + c ] = tile[tx][ty + i];
    }
}
"""


@context_dependent_memoize
def _get_shuffle_kernel(dtype):
    code = _shuffle_kernel % {
        "type": _get_register_type(dtype, memory=True)
    }
    module = SourceModule(code)
    kernel = module.get_function("dimShuffle")
    kernel.prepare("PPIIIIIIIIIIIIII")
    return kernel


def _prepare_convert_kernel(src_data, src_type, dst_data, shape, flex_data):
    # quick wrapper to convert raw float scratch data to a destination tensor
    kernel_args = [dst_data, src_data, shape[1], 1.0, flex_data]

    kernel = _get_convert_kernel(src_type)
    return [kernel, (shape[0], 1, 1), (32, 1, 1), None, ] + kernel_args


# fast axis=0 reduction kernel used for deterministic update
@context_dependent_memoize
def _get_convert_kernel(dtype):

    _convert_kernel = r"""
#include <cuda_fp16.h>

__device__ short iabs(short a)
{
    return (a < 0) ? (-a) : a;
}

__global__ void convert(short* out, const %(type)s* in, int dim,
                        float scale, int* flex_data)
{
    int offset = blockIdx.x * dim;
    int max_val = 0;

    for(int item = threadIdx.x; item < dim; item += 32)
    {
        %(type)s value = in[offset + item];
        short result = (short)(%(cvt)s(value) * scale);
        max_val = max((int)iabs(result), max_val);
        out[offset + item] = result;
    }

    atomicMax(flex_data, max_val);
}
"""
    if dtype == "f4":
        template_vals = {
            "type": "float",
            "cvt": "",
        }
    elif dtype == "f2":
        template_vals = {
            "type": "unsigned short",
            "cvt": "__half2float"
        }
    else:
        raise ValueError("Invalid conversion type")

    code = _convert_kernel % template_vals
    module = SourceModule(code)
    kernel = module.get_function("convert")
    kernel.prepare("PPIfP")
    return kernel
