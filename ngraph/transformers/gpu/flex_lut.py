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

from ngraph.transformers.gpu.float_ew2 import TensorDescriptionWrapper, FlexPtrDescription
from ngraph.transformers.gpu.kernels.cuda.lookuptable import lut_bprop_kernel_name, \
    lut_sort_kernel_name
from ngraph.transformers.gpu.lut import LUTBpropKernel


class FlexLUTBpropKernel(LUTBpropKernel):
    def __init__(self, transformer, op):
        super(FlexLUTBpropKernel, self).__init__(transformer, op)

        self.O = TensorDescriptionWrapper(self.transformer, self.O)
        self.I = TensorDescriptionWrapper(self.transformer, self.I)
        self.E = TensorDescriptionWrapper(self.transformer, self.E)

        self.flex_entry_O = self.O.flex_entry()
        self.flex_entry_I = self.I.flex_entry()
        self.flex_entry_E = self.E.flex_entry()

        self.output_flex_ids = [self.flex_entry_O.flex_id]

    def bind_buffers(self):
        super(FlexLUTBpropKernel, self).bind_buffers()
        self.flex_entry_O.allocate()
        self.flex_entry_I.allocate()
        self.flex_entry_E.allocate()

        for k in self.kernels:
            kernel, params = k
            if kernel.name == lut_bprop_kernel_name:
                maxabs_ptr = FlexPtrDescription(self.flex_entry_O)
                params.extend([maxabs_ptr,
                               self.flex_entry_O.scale,
                               self.flex_entry_I.scale,
                               self.flex_entry_E.scale])
            elif kernel.name == lut_sort_kernel_name:
                params.extend([self.flex_entry_I.scale])

        self.tensor_view_from_td(self.O.td)[:] = 0

    def bind_flex_scales(self):
        for k in self.kernels:
            kernel, params = k
            FlexPtrDescription.bind_ptr(params)

    def execute(self):
        self.word_counts.fill(0)
        for k in self.kernels:
            kernel, params = k
            kernel.prepared_async_call(*params)
