# --------------------------------------------------------------------------- classes)
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

from ngraph.transformers.gputransform import GPUTransformer, GPUKernelGroup, GPUDeviceTensor, GPUDeviceBufferStorage
from ngraph.transformers.flex2 import FlexManager

flex_verbose = False

class FlexGPUTransformer(GPUTransformer):
    """
    Flex specific functions:
    --creates flex manager
    --uses flex subclass GPUDeviceBufferStorage, which uses flex GPUDeviceTensor
    --uses flex subclass GPUKernelGroup
    """

    transformer_name = "flexgpu"

    def __init__(self, **kwargs):
        super(FlexGPUTransformer, self).__init__(**kwargs)
        self.use_flex_dtype = True  # a hack so that GPUTensorAllocator knows to use int16
        self.flex_manager = FlexManager()

    def device_buffer_storage(self, bytes, dtype, name):
        return FlexGPUDeviceBufferStorage(self, bytes, dtype, name="a_" + name)

    def gpu_kernel_group(self, name):
        return FlexGPUKernelGroup(self, name)

    def transform_ordered_ops(self, ordered_ops, name):

        ret_val = super(FlexGPUTransformer, self).transform_ordered_ops(ordered_ops, name)

        # TODO: allocate dev and host stat buffers associated with this computation?
        # tensor descriptions have already been initialized so device tensors have been created
        # create relation between computation and organization of device_buffers?
        # self.flex_manager.dev_stats.append(drv.mem_alloc(num_flex_tensors*4))

        return ret_val


class FlexGPUDeviceTensor(GPUDeviceTensor):
    """
    Scale-aware device tensor class.
    """
    def __init__(self, transformer, device_buffer, tensor_description, **kwargs):
        super(FlexGPUDeviceTensor, self).__init__(transformer, device_buffer, tensor_description,
                                              **kwargs)

        # create flex entry
        self.flex_entry = self.transformer.flex_manager.new_flex()

    @property
    def scale(self):
        return self.flex_entry.scale

    def get(self, tensor):
        tensor = super(FlexGPUDeviceTensor, self).get(tensor)
        tensor = tensor * self.scale
        return tensor

    def __setitem__(self, key, value):
        value = value / self.scale
        super(FlexGPUDeviceTensor, self).__setitem__(key, value)


class FlexGPUDeviceBufferStorage(GPUDeviceBufferStorage):

    def __init__(self, transformer, bytes, dtype, **kwargs):
        super(FlexGPUDeviceBufferStorage, self).__init__(transformer, bytes, dtype, **kwargs)

    def create_device_tensor(self, tensor_description):
        shape_str = "_".join((str(_) for _ in tensor_description.shape))
        return FlexGPUDeviceTensor(self.transformer, self, tensor_description,
                               name="v_" + tensor_description.name + "_" + shape_str)


class FlexGPUKernelGroup(GPUKernelGroup):
    """
    Performs flex setup before executing each kernel in the kernel group
    (adjust tensor scales and providing the new scales to the kernel).
    Calls autoflex algorithm after kernels execute.
    """

    def __init__(self, transformer, name):
        super(FlexGPUKernelGroup, self).__init__(transformer, name)

    def get_output_flex_ids(self):
        """
        Returns:
            list of flex ids corresponding to outputs of this kernel group
        """
        output_ids = []
        for k in self.kernels:
            assert hasattr(k, 'flex_params_info'), "{} kernel does not support flex".format(kernel)
            for _, flex_entry, is_output in k.flex_params_info:
                if is_output:
                    output_ids.append(flex_entry.flex_id)
        return output_ids

    def setup_kernel_execute(self, kernel):
        """
        Before a kernel call, tensor flex scales are adjusted
        and new values are used in kernel params
        """
        # limited kernel support for now
        # only working on elementwise and gemm while roughing out overall design
        assert hasattr(kernel, 'flex_params_info'), "{} kernel does not support flex".format(kernel)

        # TODO: name all of this something more descriptive (setup_flex_output_tensors)?

        # adjust scale of previously touched tensors (equivalent of neon flexsim output_flex)
        # TODO: is this a good assumption, that outputs of this kernel group are
        # the "previously touched tensors"?
        for flex_id in self.get_output_flex_ids():
            self.transformer.flex_manager.flex_entries[flex_id].adjust_scale()

        # set flex scale kernel parameters
        # GPUKernel flex_params_info contains a tuple record for every flex tensor
        # (position in kernel parameter list, flex entry, whether it is an output)
        # (unfortunately flex_params_info currently present for both flex and non-flex classes)
        for index, flex_entry, is_output in kernel.flex_params_info:
            scale = 1.0/flex_entry.scale if is_output else flex_entry.scale
            kernel.params[index] = scale

    def __call__(self):
        """
        Calls autoflex on touched tensors after KernelGroup call.
        """

        super(FlexGPUKernelGroup, self).__call__()

        # autoflex after calling GPUKernelGroup that is executor for computation
        touched_flex_ids = self.get_output_flex_ids()
        if flex_verbose: print "touched_flex_ids", touched_flex_ids

        if flex_verbose: print "calling autoflex, autoflexing flex_ids:", touched_flex_ids
        # autoflex sets up everything needed to adjust_scale before next use of these output tensors
        # if try to do it right away, scale for this computation is nonsensical
        self.transformer.flex_manager.autoflex(touched_flex_ids)
