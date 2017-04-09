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

from __future__ import division
import numpy as np
import imp
from third_party.warp_ctc.ctc import CTC

warp_ctc = CTC(on_device='cpu')

def ctc_cpu(acts, lbls, utt_lens, lbl_lens, grads, costs, n_threads=8):
    costs.fill(0.)
    grads.fill(0.)
    max_t, bsz, nout = acts.shape
    utt_lens = (utt_lens * max_t / 100).astype(np.int32)
    warp_ctc.bind_to_cpu(acts, 
                         lbls, 
                         utt_lens, 
                         lbl_lens, 
                         grads, 
                         costs, 
                         n_threads=n_threads)

