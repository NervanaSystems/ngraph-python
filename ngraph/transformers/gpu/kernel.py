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


def pointer_from_td(td):
    return td.value.tensor.gpudata


class GPUKernel(object):
    """
    Object which represents a single kernel that will run on the GPU.

    Arguments:
        transformer (GPUTransformer): GPU transformer containing instance of
            NervanaGPU

    Attributes:
        buffers_bound (bool): Flag indicates if GPU addresses have been bound
            to kernel parameters
        transformer (GPUTransformer): GPU transformer containing NervanaGPU
            object which is used for ops such as dot, dimshuffle, etc.
    """
    def __init__(self, transformer):
        self.buffers_bound = False
        self.transformer = transformer

    def bind_buffers(self):
        self.buffers_bound = True

    def execute(self):
        raise NotImplementedError("No execute() implemented")

    def generate_source(self, sourcefile=None):
        pass

    def compile(self, sourcefile=None):
        pass
