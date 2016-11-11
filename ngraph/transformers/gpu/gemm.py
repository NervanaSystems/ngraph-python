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

from ngraph.transformers.gpu.kernel import GPUKernel, pointer_from_td
from ngraph.transformers.gpu.float_ew2 import TensorDescriptionWrapper
from ngraph.transformers.gpu.util import _get_sm_count
from ngraph.transformers.gpu.kernels import kernel_specs
from ngraph.op_graph.axes import TensorDescription

import numpy as np


class GEMMKernel(GPUKernel):
    """
    Kernel object to execute matrix multiply on two tensors. Selects from Nervana's
    sass GEMM kernels for maxwell and later GPUs and CuBLAS for older GPUs.

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU
        op (DotOp): Graph op being transformed into this kernel

    Attributes:
        A (TensorDescriptionWrapper): Tensor for first operand
        B (TensorDescriptionWrapper): Tensor for second operand
        C (TensorDescriptionWrapper): Tensor for output
        kernel (pycuda.driver.Function): Compiled GPU kernel to execute this
            GEMM operation
        params (list): List of parameters to pass to kernel
    """
    def __init__(self, transformer, op):
        super(GEMMKernel, self).__init__(transformer)

        # Sass kernels only supported on Maxwell or newer
        if transformer.runtime.use_cudac_kernels:
            self.use_cublas = True
            self.kernel = None
            self.params = None
        else:
            self.use_cublas = False
            self._build_maxas_kernel(op)

    def _build_maxas_kernel(self, op, size=None):
        """
        Uses tensor dimensions and axis ordering to select a sass kernel and use
        maxas to compile it for later use.

        Arguments:
            op (DotOp): Graph op being transformed into this kernel
            size (str): Optional preselected tile size
        """
        # Get inputs to gemm
        C = TensorDescriptionWrapper(op.tensor_description(), 2, gemm=True)
        A, B = (TensorDescriptionWrapper(_, 2, gemm=True) for _ in op.call_info())

        # If both inputs are 1d, need to transpose one of them
        if min(A.strides) == 0 and min(B.strides) == 0:
            A.strides = tuple(reversed(A.strides))
            A.shape = tuple(reversed(A.shape))
            vector_dot = True
        else:
            vector_dot = False

        self.C = C
        self.A = A
        self.B = B

        # Kernels only support 2d tensors
        assert len(A.shape) == 2
        assert len(B.shape) == 2
        assert len(C.shape) == 2

        # one dimension must be contiguous
        assert min(A.strides) == 1 or max(A.strides) == 1
        assert min(B.strides) == 1 or max(B.strides) == 1
        assert min(C.strides) == 1 or max(C.strides) == 1 or vector_dot

        lda = max(A.strides)
        ldb = max(B.strides)
        ldc = max(C.strides)

        if A.is_trans:
            opA = 't'
            if size not in ("32x64", "16x64"):
                lda *= 8 * A.dtype.itemsize  # saves a kernel register
        else:
            opA = 'n'

        if B.is_trans:
            opB = 't'
        else:
            opB = 'n'
            if size not in ("32x64", "16x64"):
                ldb *= 8 * B.dtype.itemsize  # saves a kernel register

        op = opA + opB
        assert op != "tt"

        m = A.shape[0]
        n = B.shape[1]
        k = A.shape[1]

        assert m == C.shape[0]
        assert n == C.shape[1]
        assert k == B.shape[0]

        # Some basic tile size selection.
        # Your best bet is to benchmark your code with all 3 sizes
        # and manually fine tune the selection for each layer.
        # TODO: Perhaps I'll add an autotuning mode.
        if size is None:
            # find the shorter side
            short = min(m, n)
            # anything bigger than this just use 128
            if short < 384 - 16:
                # compute remainder of 128
                short128 = short % 128
                # if remainder is more than 112 just use 128
                if 0 < short128 < 112:
                    # to figure out when to use 64 over 32 we need to calc
                    # occupancy at 64
                    if 48 < short128 <= 64:
                        occupancy64 = short // 64
                        wide = max(m, n)
                        occupancy64 *= (wide // 128 + (wide %
                                                       128 != 0)) // _get_sm_count()
                        # 64 is only faster than 32 when occupancy is more than
                        # 1 warp per scheduler.
                        if occupancy64 > 1:
                            size = 64
                        else:
                            size = 32
                    else:
                        size = 32
                else:
                    size = 128
            # There's a large regime where 64 is faster, but it's hard to
            # characterize
            else:
                size = 128

            # match the kernel to the optimal short size but avoid not
            # implemented kernels
            if m >= n:
                if op == "nt":
                    size = 128
                sizeA, sizeB = (128, size)
            else:
                if op == "tn":
                    size = 128
                # temp till I can write these kernels (coming soon)
                elif size == 64:
                    size = 32
                sizeA, sizeB = (size, 128)

            size = "%dx%d" % (sizeA, sizeB)

        else:
            sizeA, sizeB = (int(s) for s in size.split('x'))

        gridA = m // sizeA + (m % sizeA != 0)
        gridB = n // sizeB + (n % sizeB != 0)

        k_vec = 8 if sizeA in (16, 32) or sizeB == 32 else 16

        vec_opt = None
        if op == "tn":
            if (m % 4 == 0 and n % 4 == 0 and
                    A.strides[1] % 4 == 0 and B.strides[0] % 4 == 0):
                vec_opt = ("vec",)
        elif op == "nn":
            if (k % k_vec == 0 and n % 4 == 0 and
                    A.strides[0] % k_vec == 0 and B.strides[0] % 4 == 0):
                vec_opt = ("vec",)
        elif op == "nt":
            if (k % k_vec == 0 and n % 4 == 0 and
                    A.strides[0] % k_vec == 0 and B.strides[1] % k_vec == 0):
                vec_opt = ("vec",)

        # nt and nn are more efficient with k%16==0
        if C.dtype.type is np.float16:
            clss = "hgemm"
        elif C.dtype.type is np.float32:
            clss = "sgemm"
        else:
            raise TypeError("Only floating point dot currently supported.")

        self.kernel = kernel_specs.get_kernel("_".join((clss, op, size)), vec_opt)
        self.params = [
            (1, int(gridA), int(gridB)), (self.kernel.threads, 1, 1), None,
            C.td, A.td, B.td, 1.0, 0.0, 0, int(lda), int(ldb), int(ldc),
            int(m), int(n), int(k),
            0, 0, 0, 0]

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

        super(GEMMKernel, self).bind_buffers()

    def execute(self):
        """
        Either calls into CuBLAS or runs the compiled sass GEMM kernel
        """
        if self.use_cublas:
            raise NotImplementedError("Not yet supported")
        else:
            self.kernel.prepared_async_call(*self.params)
