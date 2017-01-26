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
from ngraph.transformers.gpu.conv import ConvFpropKernel, ConvBpropKernel, ConvUpdateKernel
from ngraph.transformers.gpu.kernels import kernel_specs
from ngraph.transformers.gpu.float_ew2 import TensorDescriptionWrapper
from pycuda.compiler import SourceModule

from operator import itemgetter
import numpy as np

if sys.version_info >= (3, 0):
    from functools import reduce


class FlexConvFpropKernel(ConvFpropKernel):
    """
    Inherits all parent class ConvFpropKernel behavior except
    selects flex convolution kernel and sets up flex parameters
    """
    def __init__(self, transformer, op):
        super(FlexConvFpropKernel, self).__init__(transformer, op)
        self.alpha = 1.0
        self.beta = 0.0

    def gen_kernels(self, runtime, N, C, K, D, H, W, T, R, S, M, P, Q,
                    pad_d, pad_h, pad_w, str_d, str_h, str_w):
        self.I = TensorDescriptionWrapper(self.I, len(self.I.shape))
        self.F = TensorDescriptionWrapper(self.F, len(self.F.shape))
        self.O = TensorDescriptionWrapper(self.O, len(self.O.shape))

        self.flex_entry_I = self.I.flex_entry()
        self.flex_entry_F = self.F.flex_entry()
        self.flex_entry_O = self.O.flex_entry()

        vec_size = 4 if self.dtype.itemsize == 4 else 8

        assert N % 32 == 0, "N dim must be multiple of 32"
        assert K % vec_size == 0, "K dim must be multiple of %d" % vec_size

        if self.dtype.type is np.int16:
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
        self.relu = relu
        self.bsum = bsum

        self.all_params = (N, C, K, D, H, W, T, R, S, pad_d, pad_h, pad_w, str_d, str_h, str_w)

        self.dimI   = (C, D, H, W, N)
        self.dimF   = (C, T, R, S, K)
        self.dimFb  = (K, T, R, S, C)
        self.dimO   = (K, M, P, Q, N)
        self.dimI2  = (C*D*H*W, N)
        self.dimF2  = (C*T*R*S, K)
        self.dimF2t = (K, C*T*R*S)
        self.dimO2  = (K*M*P*Q, N)
        self.dimS   = (K, 1)
        self.sizeI  = reduce(mul, self.dimI, 1)
        self.sizeF  = reduce(mul, self.dimF, 1)
        self.sizeO  = reduce(mul, self.dimO, 1)
        self.nOut   = reduce(mul, self.MPQ, 1) * K

        # precompute some multiplications for fast constant memory access
        HW    = H*W
        DHW   = D*HW
        WN    = W*N
        HWN   = H*WN
        DHWN  = D*HWN
        RS    = R*S
        RST   = T*RS
        CRST  = C*RST
        CRSTK = K*CRST
        KRST  = K*RST
        PQ    = P*Q
        PQM   = M*PQ
        QN    = Q*N
        PQN   = P*QN
        MPQN  = M*PQN

        if CRST > 2**16:
            assert CRST  < 2**16, "Integer division is faster with 16bit numerators"

        # precompute the magic numbers and shift amounts for integer division
        magic_HW    = _magic64(HW)
        magic_W     = _magic64(W)
        magic_PQ    = _magic64(PQ)
        magic_Q     = _magic64(Q)
        magic_RST   = _magic32(CRST, RST)
        magic_RS    = _magic32(RST+32, RS)
        magic_S     = _magic32(RS+32, S)
        magic_str_w = _magic32(W + S, str_w)
        magic_str_h = _magic32(H + R, str_h)
        magic_str_d = _magic32(D + T, str_d)

        # flop count for benchmarking
        self.flops = PQM * K * N * CRST * 2.0

        tile_N   = 128 if N > 64 else 64
        grid_N   = _grid_dim(tile_N, N)
        tiles_CK = (128, 64, 32) if tile_N == 128 else (128, 64)

        ####### FPROP ###########
        self.fprop_kernels = kernel_specs.xprop_conv_kernels(
            clss, "fprop", "K", tile_N, grid_N, K, tiles_CK, PQM, RST,
            _flatten([N, K, D, H, W, WN, HWN, DHWN,
                      C, KRST, RST, RS, magic_RS, S, magic_S,
                      pad_d, pad_h, pad_w, str_d, str_h, str_w,
                      Q, PQ, QN, PQN, MPQN, magic_Q, magic_PQ]))

        # shared lookup table size
        self.fprop_lut_size = RST * 4 * 2

        # Set to 5 for the current T1000 HW config
        self.trunc_rows = 5
        flags = self.trunc_rows << 8

        self.kernels = []
        for kernel in self.fprop_kernels:
            # TODO: Populate alpha and beta parameters (in a separate loop!).
            # alpha (used to be params[6]) will be multiplied with
            self.kernels.append([
                kernel_specs.get_kernel(kernel[0]), kernel[1], kernel[2], None,
                0, self.O, self.I, self.F, 1.0, 0.0, flags,
                kernel[3]] + kernel[4])

        for kernel in self.kernels
            kernel.extend((self.flex_entry_O.ptr, 1.0))
            kernel[10] &= 0xfffffffe  # Enable output flag

        # record output flex id for autoflex
        self.output_flex_ids = [self.flex_entry_O.flex_id]
        # bind param values that depend on flex scales
        self.bind_flex_scales()

    def bind_buffers(self):
        for kernel in self.kernels:
            for index in range(len(kernel)):
                if type(kernel[index]) == TensorDescription:
                    kernel[index] = kernel[index].value.tensor.gpudata

        super(FlexConvFpropKernel, self).bind_buffers()

    def bind_flex_scales(self):
        scaleAB = self.flex_entry_I.scale * self.flex_entry_F.scale
        scaleC = self.flex_entry_O.scale
        alpha = self.alpha * scaleAB
        beta = self.beta * scaleC

        for kernel in self.kernels:
            kernel[8] = alpha
            kernel[9] = beta
            kernel[-1] = 1. / scaleC

    def execute(self):
        for kernel in self.kernels:
            kernel[0].prepared_async_call(*kernel[1:], shared_size=self.fprop_lut_size)


class FlexConvBpropKernel(ConvBpropKernel):
    """
    Inherits all parent class ConvBpropKernel behavior except
    selects flex convolution kernel and sets up flex parameters
    """
    def __init__(self, transformer, op):
        super(FlexConvBpropKernel, self).__init__(transformer, op)
        self.alpha = 1.0
        self.beta = 0.0

    def gen_kernels(self, runtime, N, C, K, D, H, W, T, R, S, M, P, Q,
                    pad_d, pad_h, pad_w, str_d, str_h, str_w):
        self.I = TensorDescriptionWrapper(self.I, len(self.I.shape))
        self.F = TensorDescriptionWrapper(self.F, len(self.F.shape))
        self.O = TensorDescriptionWrapper(self.O, len(self.O.shape))

        self.flex_entry_I = self.I.flex_entry()
        self.flex_entry_F = self.F.flex_entry()
        self.flex_entry_O = self.O.flex_entry()

        F_size = np.prod(self.F.shape) * self.F.dtype.itemsize
        O_size = np.prod(self.O.shape) * self.O.dtype.itemsize

        vec_size = 4 if self.dtype.itemsize == 4 else 8

        assert N % 32 == 0, "N dim must be multiple of 32"
        assert K % vec_size == 0, "K dim must be multiple of %d" % vec_size

        if self.dtype.type is np.int16:
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
        self.relu = relu
        self.bsum = bsum

        self.all_params = (N, C, K, D, H, W, T, R, S, pad_d, pad_h, pad_w, str_d, str_h, str_w)

        self.dimI   = (C, D, H, W, N)
        self.dimF   = (C, T, R, S, K)
        self.dimFb  = (K, T, R, S, C)
        self.dimO   = (K, M, P, Q, N)
        self.dimI2  = (C*D*H*W, N)
        self.dimF2  = (C*T*R*S, K)
        self.dimF2t = (K, C*T*R*S)
        self.dimO2  = (K*M*P*Q, N)
        self.dimS   = (K, 1)
        self.sizeI  = reduce(mul, self.dimI, 1)
        self.sizeF  = reduce(mul, self.dimF, 1)
        self.sizeO  = reduce(mul, self.dimO, 1)
        self.nOut   = reduce(mul, self.MPQ, 1) * K

        # precompute some multiplications for fast constant memory access
        HW    = H*W
        DHW   = D*HW
        WN    = W*N
        HWN   = H*WN
        DHWN  = D*HWN
        RS    = R*S
        RST   = T*RS
        CRST  = C*RST
        CRSTK = K*CRST
        KRST  = K*RST
        PQ    = P*Q
        PQM   = M*PQ
        QN    = Q*N
        PQN   = P*QN
        MPQN  = M*PQN

        if CRST > 2**16:
            assert CRST  < 2**16, "Integer division is faster with 16bit numerators"

        # precompute the magic numbers and shift amounts for integer division
        magic_HW    = _magic64(HW)
        magic_W     = _magic64(W)
        magic_PQ    = _magic64(PQ)
        magic_Q     = _magic64(Q)
        magic_RST   = _magic32(CRST, RST)
        magic_RS    = _magic32(RST+32, RS)
        magic_S     = _magic32(RS+32, S)
        magic_str_w = _magic32(W + S, str_w)
        magic_str_h = _magic32(H + R, str_h)
        magic_str_d = _magic32(D + T, str_d)

        # flop count for benchmarking
        self.flops = PQM * K * N * CRST * 2.0

        tile_N   = 128 if N > 64 else 64
        grid_N   = _grid_dim(tile_N, N)
        tiles_CK = (128, 64, 32) if tile_N == 128 else (128, 64)

        ####### BPROP ###########
        if C < 16 or C % vec_size != 0:
            # special kernel for deconv into first layer
            kernel_name = "%s_bprop_C1_N64" % clss

            grid  = (PQM, _grid_dim(32, CRST), _grid_dim(64, N))
            block = (32, 1, 1)

            self.bprop_kernels = [[kernel_name, grid, block, 0, _flatten([
                N, K, D, H, W, WN, HWN, DHWN,
                C, CRST, RST, magic_RST, RS, magic_RS, S, magic_S,
                pad_d, pad_h, pad_w, str_d, str_h, str_w,
                Q, PQ, QN, PQN, MPQN, magic_Q, magic_PQ,
                CRST*8*self.dtype.itemsize, MPQN*8*self.dtype.itemsize])]]

            # generate the kernel args for transpose CRST,K => K,CRST
            self.shuffle_args = [CRST, K]
            gridX   = (K    >> 5) + (K    & 31 != 0)
            gridY   = (CRST >> 5) + (CRST & 31 != 0)
            self.shuffle_grid   = (gridX, gridY, 1)
            self.shuffle_block  = (32, 8, 1)
            self.bprop_zero     = self.sizeI * self.dtype.itemsize
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
                RST*K, RS*K, S*K, K,
                RST*C, RS*C, S*C, C,
                RS, magic_RS, S, magic_S])
            gridX = (K >> 5) + (K & 31 != 0)
            gridY = (C >> 5) + (C & 31 != 0)
            self.shuffle_grid   = (gridX, gridY, RST)
            self.shuffle_block  = (32, 8, 1)
            self.bprop_zero     = 0
            self.bprop_lut_size = RST * 4 * 2

        # Set to 5 for the current T1000 HW config
        self.trunc_rows = 5
        flags = self.trunc_rows << 8

        F_data = runtime.scratch_buffer(B.size)
        if self.bprop_zero:
            Out = runtime.scratch_buffer(O_size, offset=F_size)
        else:
            Out = self.O

        self.kernels = []
        for kernel in self.bprop_kernels:
            # TODO: Populate alpha and beta parameters (in a separate loop!).
            # alpha (used to be params[6]) will be multiplied with
            self.kernels.append([
                kernel_specs.get_kernel(kernel[0]), kernel[1], kernel[2], None,
                0, Out, self.I, F_data, 1.0, 0.0, flags, kernel[3]] + kernel[4])

        for kernel in self.kernels
            kernel.extend((self.flex_entry_O.ptr, 1.0))
            kernel[10] &= 0xfffffffe  # Enable output flag

        # record output flex id for autoflex
        self.output_flex_ids = [self.flex_entry_O.flex_id]
        # bind param values that depend on flex scales
        self.bind_flex_scales()
        
        if self.bprop_zero:
            shuffle_kernel = _get_transpose_kernel(self.F.dtype.str[1:])
            self.convert_type = "f2"  # source type
        else:
            shuffle_kernel = _get_shuffle_kernel(self.F.dtype.str[1:])  # can point to transpose or dimshuffle kernel
        shuffle_args = [layer.shuffle_grid, self.shuffle_block, None,
                        F_data, self.F] + self.shuffle_args
        shuffle_kernel = [shuffle_kernel] + shuffle_args
        self.kernels.append(shuffle_kernel)

    def bind_buffers(self):
        for kernel in self.kernels:
            for index in range(len(kernel)):
                if type(kernel[index]) == TensorDescription:
                    kernel[index] = kernel[index].value.tensor.gpudata

        super(FlexConvBpropKernel, self).bind_buffers()

    def bind_flex_scales(self):
        scaleAB = self.flex_entry_I.scale * self.flex_entry_F.scale
        scaleC = self.flex_entry_O.scale
        alpha = self.alpha * scaleAB
        beta = self.beta * scaleC

        for kernel in self.kernels:
            kernel[8] = alpha
            kernel[9] = beta
            kernel[-1] = 1. / scaleC

    def execute(self):
        for kernel in self.kernels:
            kernel[0].prepared_async_call(*kernel[1:], shared_size=self.bprop_lut_size)


class FlexConvUpdateKernel(ConvUpdateKernel):
    """
    Inherits all parent class ConvUpdateKernel behavior except
    selects flex convolution kernel and sets up flex parameters
    """
    def __init__(self, transformer, op):
        raise NotImplementedError


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

    code = _transpose_kernel % _ew_types[dtype]
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

    code = _shuffle_kernel % _ew_types[dtype]
    module = SourceModule(code)
    kernel = module.get_function("dimShuffle")
    kernel.prepare("PPIIIIIIIIIIIIII")
    return kernel
