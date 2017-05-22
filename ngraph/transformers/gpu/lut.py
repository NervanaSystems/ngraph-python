# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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

from ngraph.op_graph.axes import TensorDescription
from ngraph.transformers.gpu.kernel import GPUKernel
from ngraph.transformers.gpu.kernels.cuda import lookuptable

from pycuda.gpuarray import empty
import numpy as np


class LUTBpropKernel(GPUKernel):
    """
    Kernel object to execute lookup table backward propagation. Selects from Nervana's
    cuda lookup table kernels.

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU
        op (update_lut): Graph op being transformed into this kernel
    """
    def __init__(self, transformer, op):
        super(LUTBpropKernel, self).__init__(transformer)
        self.op = op

        # Hard coding for now, non-deterministic is faster but difficult to reproduce
        # or debug. Deterministic kernels are fast enough and LUT layer tends to be
        # a small percentage of execution time.
        self.deterministic = True

        (self.E, self.I) = (_ for _ in op.call_info())
        self.O = op.tensor_description()
        pad_idx = op.pad_idx
        lut_axis = op.lut_axis
        # Only supported when reads are contiguous
        assert (lut_axis == 0)

        embedding_dim = self.O.shape[1]
        vocab_size = self.O.shape[0]
        nin = self.E.shape[0]

        if pad_idx is None:
            pad_idx = int(-1)

        self.kernels = []

        if self.deterministic:
            self.index_buffer = empty((nin,), dtype=np.int32)
            self.offset_buffer = empty((nin,), dtype=np.int32)
            self.word_counts = empty((max(512, vocab_size) + 512,), dtype=np.int32)

            for kernel_id in range(5):
                threads = 512
                if kernel_id in [1, 3]:
                    blocks = vocab_size // (threads * 2)
                    if vocab_size % (threads * 2):
                        blocks = blocks + 1
                elif kernel_id == 2:
                    blocks = 1
                else:
                    blocks = nin // threads
                    if nin % threads:
                        blocks = blocks + 1

                params = [(blocks, 1, 1), (threads, 1, 1), None,
                          self.I, self.index_buffer.gpudata, self.offset_buffer.gpudata,
                          self.word_counts.gpudata, max(512, vocab_size), nin]
                kernel = lookuptable._get_sorting_kernel(kernel_id, threads, self.I.dtype)
                self.kernels.append((kernel, params))

            threads = 32
            blocks = nin

            params = [(blocks, 1, 1), (threads, 1, 1), None,
                      self.I, self.index_buffer.gpudata, self.O, self.E,
                      nin, embedding_dim, vocab_size, pad_idx]

            kernel = lookuptable._get_lut_bprop_kernel(self.E.dtype, self.I.dtype, True)
            self.kernels.append((kernel, params))

    def bind_buffers(self):
        """
        Gets allocated tensors for input and output feature maps.
        Allocates a scratch tensor for argmax indices if the op is max pooling
        since this is required for bprop. Builds a final list of parameters to
        pass to the kernel.
        """
        for k in self.kernels:
            for index in range(len(k[1])):
                if isinstance(k[1][index], TensorDescription):
                    k[1][index] = self.pointer_from_td(k[1][index])

        super(LUTBpropKernel, self).bind_buffers()

    def execute(self):
        """
        Executes the pooling kernel.
        """
        self.tensor_view_from_td(self.O).tensor.fill(0)
        self.word_counts.fill(0)
        for k in self.kernels:
            kernel, params = k
            kernel.prepared_async_call(*params)
