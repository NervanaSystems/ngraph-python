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

from ngraph.transformers.gpu.float_ew2 import FlexPtrDescription

from ngraph.transformers.gpu.pool import PoolFpropKernel, PoolBpropKernel


def convert_to_flex(kernel):
    class FlexPoolKernel(kernel):
        supported_types = ['flex']

        def __init__(self, transformer, op):
            super(FlexPoolKernel, self).__init__(transformer, op)
            self.output_flex_ids = []
            self.in_op = self.op.args[0]
            self.out_op = self.op

        def bind_buffers(self):
            super(FlexPoolKernel, self).bind_buffers()
            maxabs_ptr = FlexPtrDescription(self.transformer.get_op_tensor(self.op).flex_entry)
            self.params.extend([maxabs_ptr])

        def bind_flex_scales(self):
            self.transformer.get_op_tensor(self.op).flex_entry.allocate()
            FlexPtrDescription.bind_ptr(self.params)

    return FlexPoolKernel


class FlexPoolFpropKernel(convert_to_flex(PoolFpropKernel)):
    """
    Flex version of PoolFpropKernel
    """


class FlexPoolBpropKernel(convert_to_flex(PoolBpropKernel)):
    """
    Flex version of PoolBpropKernel
    """
