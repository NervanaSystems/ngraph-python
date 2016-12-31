# ---------------------------------------------------------------------------
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

from ngraph.transformers.gputransform import GPUTransformer, GPUKernelGroup
from ngraph.transformers.gputransform import GPUDeviceTensor, GPUDeviceBufferStorage
from ngraph.transformers.gputransform import ElementWiseKernel
from ngraph.transformers.passes.flexpass import FlexPass
from ngraph.transformers.gpu.float_ew2 import FlexScaleDescription
from autoflex.flexgpu import GPUFlexManager, GPUFlex, gpu_bind_flex_params


# create and attach bind_flex_scales method to EW kernel (avoid editing gputransform)
def _ew_bind_flex_scales(kernel):
    for index, flex_scale_desc in kernel.flex_scale_info:
        scale = flex_scale_desc.flex_entry.scale
        scale = 1.0 / scale if flex_scale_desc.is_output else scale
        kernel.params[index] = scale
ElementWiseKernel.bind_flex_scales = _ew_bind_flex_scales


class FlexGPUTransformer(GPUTransformer):
    """
    Flex specific functions:
    --creates flex manager
    --uses flex subclass GPUDeviceBufferStorage, which uses flex GPUDeviceTensor
    --uses flex subclass GPUKernelGroup
    """

    transformer_name = "gpuflex"

    # set global override tolerances for unit tests
    fixed_point_res = GPUFlexManager.fixed_point_resolution()

    # TODO haven't investigated how these should be set, start with small tol
    default_rtol = 1e-05
    default_atol = 20 * fixed_point_res

    def __init__(self, **kwargs):

        super(FlexGPUTransformer, self).__init__(**kwargs)

        # flex passes for setting Op dtypes to flex
        # TODO: verify ClearTensorDescription pass not needed with core graph team
        # self.register_graph_pass(ClearTensorDescriptions())
        self.register_graph_pass(FlexPass())

        # flex manager manages autoflex mechanics
        self.flex_manager = GPUFlexManager(fixed_point=False, verbose=True)

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

    def storage_dtype(self, dtype):
        if isinstance(dtype, GPUFlex):
            return dtype.storage_dtype
        else:
            raise NotImplementedError


class FlexGPUDeviceTensor(GPUDeviceTensor):
    """
    Scale-aware device tensor class.
    """
    def __init__(self, transformer, device_buffer, tensor_description, **kwargs):
        super(FlexGPUDeviceTensor, self).__init__(transformer,
                                                  device_buffer,
                                                  tensor_description,
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

    def compile_all(self):
        """
        subclass deals with ElementWiseKernel flex interface here in order to
        isolate from gputransform
        """

        super(FlexGPUKernelGroup, self).compile_all()

        self._create_output_flex_ids()
        self._create_ew_flex_scale_info()

    def _create_output_flex_ids(self):
        """
        TODO: cleanup docstring, esp about EW

        This method creates output_flex_ids attribute for the kernel group FlexGPUKernelGroup
        It also creates output_flex_ids for ElementWiseKernel to avoid modifying gputransform

        Kernels that actually modify tensor values should have output_flex_ids attribute
        Kernel group output_flex_ids attribute is the set of all output_flex_ids of
        its component kernels

        "output" tensors: tensors that will be modified by this kernel group
        """
        # TODO: explicitly list kernels for now to catch any missing
        from ngraph.transformers.gputransform import FillKernel, DimShuffleKernel, RngFillKernel
        no_output_id_kernels = (FillKernel, DimShuffleKernel, RngFillKernel)

        # create output_flex_ids for overall kernel group and
        # create output_flex_ids for kernels that don't already have them
        group_output_ids = []
        for kernel in self.kernels:
            # have to create output_flex_ids here for EW
            if isinstance(kernel, ElementWiseKernel):
                # look for FlexScaleDescription.is_output
                # at compile time scales have not been bound yet so this still exists
                kernel_output_ids = []
                for p in kernel.params:
                    if isinstance(p, FlexScaleDescription) and p.is_output:
                        kernel_output_ids.append(p.flex_entry.flex_id)
                kernel.output_flex_ids = kernel_output_ids
            elif isinstance(kernel, no_output_id_kernels):
                kernel.output_flex_ids = []

            # now add kernel output_flex_ids to kernel group list of output ids
            group_output_ids.extend(kernel.output_flex_ids)

        # kernel group output_flex_ids is combined list over all kernels
        self.output_flex_ids = group_output_ids

    def _create_ew_flex_scale_info(self):
        """
        TODO: cleanup docstring
        Set up EW bind_flex_scales method
        Avoid modifying gputransform
        """

        # EW store index and description of flex scale params that need to be changed each call
        for kernel in self.kernels:
            if isinstance(kernel, ElementWiseKernel):
                scale_info = [(i, p) for i, p in enumerate(kernel.params)
                              if isinstance(p, FlexScaleDescription)]
                kernel.flex_scale_info = scale_info

    def setup_kernel_execute(self, kernel):
        """
        Before a kernel call, flex tensor scales are adjusted
        and new values are bound to kernel params
        """

        # both kernel group and component kernels have output_flex_ids
        # iterative over output_flex_ids specific to this kernel
        for flex_id in kernel.output_flex_ids:
            # adjust scale of previously touched tensors
            flex_entry = self.transformer.flex_manager.flex_entries[flex_id]
            flex_entry.manage_before_computation(kernel)

        # TODO: move this inside manage_before_computation?
        # bind flex scale kernel parameters
        gpu_bind_flex_params(kernel)

    def __call__(self):
        """
        Calls autoflex on touched tensors after KernelGroup call.
        """

        super(FlexGPUKernelGroup, self).__call__()

        # autoflex after calling GPUKernelGroup that is executor for computation
        flex_manager = self.transformer.flex_manager
        if flex_manager.fixed_point is False:

            # autoflex
            # set up everything needed before next use of these output tensors

            if flex_manager.verbose:
                print "autoflexing flex_ids:", self.output_flex_ids

            self.transformer.flex_manager.autoflex(self.output_flex_ids)
