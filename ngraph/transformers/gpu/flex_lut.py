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

        kernel, params = self.kernels[-1]
        maxabs_ptr = FlexPtrDescription(self.flex_entry_O)
        params.extend([maxabs_ptr])

        self.flex_entry_O.allocate()
        self.flex_entry_I.allocate()
        self.flex_entry_E.allocate()

        scale_o = self.flex_entry_O.scale
        scale_i = self.flex_entry_I.scale
        scale_e = self.flex_entry_E.scale

        kernel, params = self.kernels[-1]
        params.extend([scale_o, scale_i, scale_e])

        for k in self.kernels[:-1]:
            kernel, params = k
            params.extend([scale_i])

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
