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

from __future__ import print_function
from ngraph.transformers.gpu.kernel import GPUKernel
import numpy as np
from third_party.warp_ctc.ctc import CTC


class CTCKernel(GPUKernel):
    def __init__(self, transformer, op):
        super(CTCKernel, self).__init__(transformer)

        self.warp_ctc = CTC(on_device='gpu')
        self.at_runtime = self.transformer.runtime
        self.stream = self.at_runtime.stream
        self.costs = op.tensor_description()
        (self.activs,
         self.lbls,
         self.uttlens_pct,
         self.lbl_lens,
         self.grads) = (_ for _ in op.call_info())
        self.max_t, self.bsz, self.nout = self.activs.axes.lengths

    def bind_buffers(self):
        self.activs = self.activs.value.tensor
        self.lbls = self.lbls.value.tensor
        self.uttlens_pct = self.uttlens_pct.value.tensor
        self.lbl_lens = self.lbl_lens.value.tensor
        self.grads = self.grads.value.tensor
        self.costs = self.costs.value.tensor

        super(CTCKernel, self).bind_buffers()

    def execute(self):
        self.grads.fill(0.)
        self.costs.fill(0.)

        warp_utt_lens = (self.uttlens_pct.get().ravel()
                         * self.max_t / 100.).astype(np.int32)
        warp_lbls = self.lbls.get().ravel().astype(np.int32)
        warp_lbl_lens = self.lbl_lens.get().ravel().astype(np.int32)

        scratch_size = self.warp_ctc.get_gpu_workspace_size(warp_lbl_lens,
                                                            warp_utt_lens,
                                                            self.nout,
                                                            self.bsz)
        self.at_runtime.set_scratch_size(scratch_size)
        workspace = self.at_runtime.scratch_buffer(scratch_size)

        self.warp_ctc.bind_to_gpu(self.activs,
                                  self.grads,
                                  warp_lbls,
                                  warp_lbl_lens,
                                  warp_utt_lens,
                                  self.costs,
                                  workspace,
                                  scratch_size,
                                  self.stream)
