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

from ngraph.transformers.gpu.kernel import GPUKernel

import numpy as np
import os
import numpy.ctypeslib as npct
import ctypes as ct


class CTCKernel(GPUKernel):
    def __init__(self, transformer, op):
        super(CTCKernel, self).__init__(transformer)

        basepath = os.path.join(os.path.dirname(__file__), "..", "..", "..")
        temp_loc = os.path.join("examples", "deepspeech", "src", "libwarpctc.so")
        libpath = os.path.join(basepath, temp_loc)
        assert os.path.isfile(libpath), "libwarpctc.so not found.  Run make"

        self.ctclib = npct.load_library(libpath, "")
        self.at_runtime = self.transformer.runtime
        self.stream = self.at_runtime.stream
        self.costs = op.tensor_description()
        (self.activs,
         self.lbls,
         self.uttlens_pct,
         self.lbl_lens,
         self.grads) = (_ for _ in op.call_info())
        self.max_t, self.bsz, self.nout = self.activs.axes.lengths
        self.utt_lens = np.zeros(self.bsz, dtype=np.int32)

    def bind_buffers(self):
        self.activs = self.activs.value.tensor
        self.lbls = self.lbls.value.tensor
        self.uttlens_pct = self.uttlens_pct.value.tensor
        self.lbl_lens = self.lbl_lens.value.tensor
        self.grads = self.grads.value.tensor
        self.costs = self.costs.value.tensor
        super(CTCKernel, self).bind_buffers()

    def execute(self):
        if self.stream is None:
            stream_buf = ct.cast(self.stream, ct.c_void_p)
        else:
            stream_buf = ct.cast(self.stream.handle, ct.c_void_p)

        # TODO: figure out how to do this conversion outside the op
        self.utt_lens[:] = (self.uttlens_pct.get() * self.max_t / 100.).astype(np.int32)
        self.grads.fill(0.)
        self.costs.fill(0.)

        self.ctclib.get_workspace_size_gpu.restype = int
        self.ctclib.get_workspace_size_gpu.argtypes = [npct.ndpointer(dtype=np.int32,
                                                                      ndim=1),
                                                       npct.ndpointer(dtype=np.int32,
                                                                      ndim=1),
                                                       ct.c_int,
                                                       ct.c_int,
                                                       ct.c_void_p]

        scratch_size = self.ctclib.get_workspace_size_gpu(np.array(self.lbl_lens.get(),
                                                                   dtype=np.int32),
                                                          self.utt_lens,
                                                          self.nout, self.bsz,
                                                          stream_buf)

        self.at_runtime.set_scratch_size(scratch_size)
        workspace = self.at_runtime.scratch_buffer(scratch_size)

        self.ctclib.compute_ctc_loss_gpu.restype = int
        self.ctclib.compute_ctc_loss_gpu.argtypes = [ct.POINTER(ct.c_float),
                                                     ct.POINTER(ct.c_float),
                                                     npct.ndpointer(
                                                         dtype=np.int32, ndim=1),
                                                     npct.ndpointer(
                                                         dtype=np.int32, ndim=1),
                                                     npct.ndpointer(
                                                         dtype=np.int32, ndim=1),
                                                     ct.c_int,
                                                     ct.c_int,
                                                     ct.POINTER(ct.c_float),
                                                     ct.POINTER(ct.c_char),
                                                     ct.c_void_p]

        acts_buf = ct.cast(int(self.activs.gpudata), ct.POINTER(ct.c_float))
        grads_buf = ct.cast(int(self.grads.gpudata), ct.POINTER(ct.c_float))
        costs_buf = ct.cast(int(self.costs.gpudata), ct.POINTER(ct.c_float))
        workspace_buf = ct.cast(workspace, ct.POINTER(ct.c_char))

        status = self.ctclib.compute_ctc_loss_gpu(acts_buf,
                                                  grads_buf,
                                                  np.array(self.lbls.get(),
                                                           dtype=np.int32),
                                                  np.array(self.lbl_lens.get(),
                                                           dtype=np.int32),
                                                  self.utt_lens,
                                                  self.nout,
                                                  self.bsz,
                                                  costs_buf,
                                                  workspace_buf,
                                                  stream_buf)

        assert status is 0, "Warp-CTC run failed"
