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

    def pointer_from_td(self, td):
        """
        Gets a GPU address from an allocated TensorDescription

        Arguments:
            td (TensorDescription): Tensor to get the address of

        Returns: A GPU address (int or pycuda.DeviceAllocation)
        """
        return self.transformer.get_tensor_description_tensor_view(td).tensor.gpudata

    def tensor_view_from_td(self, td):
        return self.transformer.get_tensor_description_tensor_view(td)

    def bind_buffers(self):
        """
        Binds GPU addresses of buffers to the kernel parameters. When kernels
        and initial parameters are generated, tensors have not yet been
        allocated so a placeholder is used for the memory addresses. This must
        be called before the first kernel run to bind the tensor addresses in
        GPU memory to the kernel parameters.
        """
        self.buffers_bound = True

    def execute(self):
        """
        Runs the kernel
        """
        raise NotImplementedError("No execute() implemented")

    def generate_source(self, sourcefile=None):
        """
        Called when the kernel is added to the kernel group. The group's
        shared sourcefile is passed so that all kernels in a group can
        add code to the same file.

        Arguments:
            sourcefile (CudaSourceFile): Source file to generate the kernel's
                code into
        """
        pass

    def compile(self, sourcefile=None):
        """
        Called after NVCC has been run on the sourcefile passed to generate_source
        to create a binary. Kernels can query the sourcefile for function pointers
        to the compiled kernel in this function.

        Arguments:
            sourcefile (CudaSourceFile): Source file that was passed to generate_source
        """
        pass
