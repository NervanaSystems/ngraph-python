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

from __future__ import print_function
from __future__ import division
import numpy as np

from ngraph.transformers.base import UnsupportedTransformerException
from ngraph.transformers.gpu.flex_lut import FlexLUTBpropKernel
from ngraph.transformers.passes.flexfusion import FlexFusion

try:
    from ngraph.flex import GPUFlexManager, GPUFlex, gpuflex16
except ImportError:
    raise UnsupportedTransformerException("autoflex package not installed")

from ngraph.op_graph.op_graph import Op, Fill, RngOp, TensorSizeOp, AssignOp
from ngraph.op_graph.pooling import PoolingOp, BpropPoolOp
from ngraph.op_graph.convolution import ConvolutionOp, bprop_conv, update_conv
from ngraph.transformers.gputransform import GPUTransformer, GPUKernelGroup
from ngraph.transformers.gputransform import GPUDeviceTensor, GPUDeviceBufferStorage
from ngraph.transformers.gputransform import ElementWiseKernel
from ngraph.transformers.gpu.flex_conv import FlexConvFpropKernel, FlexConvBpropKernel, \
    FlexConvUpdateKernel
from ngraph.transformers.gpu.flex_pool import FlexPoolFpropKernel, FlexPoolBpropKernel
from ngraph.transformers.gpu.tensor_ops import FlexFillKernel, FlexRngFillKernel, FlexAssignKernel
from ngraph.transformers.passes.flexpass import FlexDtypePass, FlexPropagateEntryPass, \
    ClearTensorDescriptions, FlexStackOpPass
from ngraph.transformers.gpu.float_ew2 import CudaSourceFile, FlexScaleDescription, \
    FlexPtrDescription
from ngraph.flex.names import flex_gpu_transformer_name
from ngraph.util.generics import generic_method
from ngraph.op_graph.lookuptable import update_lut

# kernels that do not require flex integration
# non_flex_kernels: output_flex_ids set to empty list
from ngraph.transformers.gputransform import DimShuffleKernel
non_flex_kernels = (DimShuffleKernel, )


# create and attach bind_flex_scales method to EW kernel
# done this way to avoid editing gputransform
def _ew_bind_flex_scales(kernel):
    for index, flex_scale_desc in kernel.flex_scale_info:
        scale = flex_scale_desc.flex_entry.scale
        scale = 1.0 / scale if flex_scale_desc.is_output else scale
        kernel.params[index] = scale
    FlexPtrDescription.bind_ptr(kernel.params)


ElementWiseKernel.bind_flex_scales = _ew_bind_flex_scales


class FlexGPUTransformer(GPUTransformer):
    """
    Flex specific functions:
     - creates flex manager
     - uses flex subclass GPUDeviceBufferStorage, which uses flex GPUDeviceTensor
     - uses flex subclass GPUKernelGroup
    """

    transformer_name = flex_gpu_transformer_name

    # set global override tolerances for unit tests
    fixed_point_res = GPUFlexManager.fixed_point_resolution()

    default_rtol = 1e-02
    atol_precision_multiplier = 8

    def __init__(self, fixed_point=False, flex_verbose=False, collect_flex_data=False, **kwargs):

        super(FlexGPUTransformer, self).__init__()
        self.fixed_point = fixed_point
        self.collect_flex_data = collect_flex_data

        # flex passes for setting Op dtypes to flex
        self.register_graph_pass(FlexFusion(), 0)
        self.register_graph_pass(ClearTensorDescriptions(transformer=self))
        self.register_graph_pass(FlexDtypePass())

        # flex manager manages autoflex mechanics
        self.flex_manager = GPUFlexManager(fixed_point=fixed_point,
                                           verbose=flex_verbose)

    @classmethod
    def get_default_tolerance(cls, desired):
        scale = gpuflex16.get_scale(np.max(abs(desired)))
        atol = scale * FlexGPUTransformer.atol_precision_multiplier
        rtol = FlexGPUTransformer.default_rtol
        return atol, rtol

    def set_output_statistics_file(self, statistics_file):
        if self.collect_flex_data:
            self.set_flex_data_file(statistics_file)

    def save_output_statistics_file(self):
        if self.collect_flex_data:
            self.flex_manager.save_diagnostic_data()

    def set_flex_data_file(self, data_file):
        self.flex_manager.set_h5py_file(data_file)

    def device_buffer_storage(self, bytes, dtype, name):
        return FlexGPUDeviceBufferStorage(self, bytes, dtype, name="a_" + name)

    def add_flex_id_to_ops(self):
        for op in self.ops:
            if op.is_tensor_op:
                base = op.forwarded.tensor_description().base
                tensor = self.op_tensors.get(base, None)
                if hasattr(tensor, 'flex_entry'):
                    op.metadata['flex_id'] = tensor.flex_entry.flex_id
                    base.op.metadata['flex_id'] = tensor.flex_entry.flex_id

    def start_transform_allocate(self):
        super(FlexGPUTransformer, self).start_transform_allocate()
        self.add_flex_id_to_ops()

        # VizPass(subgraph_attr="flex_id").wrapped_do_pass(ops=self.ops)

    def gpu_kernel_group(self, name):
        return FlexGPUKernelGroup(self, name)

    def finish_transform_allocate(self):
        super(FlexGPUTransformer, self).finish_transform_allocate()

        FlexPropagateEntryPass(transformer=self).do_pass(ops=self.ops)
        FlexStackOpPass(transformer=self).do_pass(ops=self.ops)

    def transform_ordered_ops(self, computation, ordered_ops, name):
        ret_val = super(FlexGPUTransformer, self).transform_ordered_ops(
            computation, ordered_ops, name)

        # device memory allocation after drv init
        self.flex_manager.allocate()

        return ret_val

    def storage_dtype(self, dtype):
        if isinstance(dtype, GPUFlex):
            return dtype.storage_dtype
        else:
            # TODO
            raise NotImplementedError


class FlexGPUDeviceTensor(GPUDeviceTensor):
    """
    Flex scale-aware device tensor class.
    """
    def __init__(self, transformer, device_buffer, tensor_description, **kwargs):
        super(FlexGPUDeviceTensor, self).__init__(transformer,
                                                  device_buffer,
                                                  tensor_description,
                                                  **kwargs)

    @property
    def scale(self):
        return self.flex_entry.scale

    @property
    def flex_entry(self):
        return self.device_buffer.flex_entry

    def get(self, tensor):
        tensor = super(FlexGPUDeviceTensor, self).get(tensor)
        tensor = tensor * self.scale
        return tensor

    def __setitem__(self, key, value):

        # flex management
        # though not a kernel, setitem modifies tensor values
        self.flex_entry.manage_before_computation(value)

        # store integer representation
        value = value / self.scale

        # check for overflow and clip
        bits = self.flex_entry.dtype.storage_bits - 1
        maxval = 2**bits - 1
        minval = -2**bits
        if isinstance(value, (int, float)):
            value = min(value, maxval) if value >= 0 else max(value, minval)
        else:
            value[value > maxval] = maxval
            value[value < minval] = minval

        # set modified values
        super(FlexGPUDeviceTensor, self).__setitem__(key, value)

        # flex management
        maxabs = int(np.amax(np.absolute(value)))
        self.flex_entry.manage_after_computation(maxabs,
                                                 self.transformer.flex_manager.autoflex_count)


class FlexGPUDeviceBufferStorage(GPUDeviceBufferStorage):

    def __init__(self, transformer, bytes, dtype, **kwargs):
        super(FlexGPUDeviceBufferStorage, self).__init__(transformer, bytes, dtype, **kwargs)

        # create flex entry
        self.flex_entry = self.transformer.flex_manager.make_flex_entry(name=self.name)

    def create_device_tensor(self, tensor_description):
        name = self.get_tensor_name(tensor_description)
        return FlexGPUDeviceTensor(self.transformer, self, tensor_description, name=name)


class FlexGPUKernelGroup(GPUKernelGroup):
    """
    Performs flex setup before executing each kernel in the kernel group
    (adjust tensor scales and providing the new scales to the kernel).
    Calls autoflex algorithm after kernels execute.
    """

    def __init__(self, transformer, name):
        super(FlexGPUKernelGroup, self).__init__(transformer, name)

    def make_cuda_source_file(self):
        return CudaSourceFile(self.name, gen_flex=True, retain_file=False)

    @generic_method(Op)
    def add_kernel(self, op):
        super(FlexGPUKernelGroup, self).add_kernel(op)

    @add_kernel.on_type(AssignOp)
    def add_kernel(self, op):
        self.kernels.append(FlexAssignKernel(self.transformer,
                                             op.call_info()[0], op.call_info()[1]))

    @add_kernel.on_type(ConvolutionOp)
    def add_kernel(self, op):
        self.kernels.append(FlexConvFpropKernel(self.transformer, op))

    @add_kernel.on_type(bprop_conv)
    def add_kernel(self, op):
        self.kernels.append(FlexConvBpropKernel(self.transformer, op))

    @add_kernel.on_type(update_conv)
    def add_kernel(self, op):
        self.kernels.append(FlexConvUpdateKernel(self.transformer, op))

    @add_kernel.on_type(Fill)
    def add_kernel(self, op):
        self.kernels.append(FlexFillKernel(self.transformer, op.call_info()[0], op.scalar))

    @add_kernel.on_type(RngOp)
    def add_kernel(self, op):
        self.kernels.append(FlexRngFillKernel(self.transformer,
                                              op.tensor_description(),
                                              op.distribution,
                                              op.params))

    @add_kernel.on_type(PoolingOp)
    def add_kernel(self, op):
        self.kernels.append(FlexPoolFpropKernel(self.transformer, op))

    @add_kernel.on_type(BpropPoolOp)
    def add_kernel(self, op):
        self.kernels.append(FlexPoolBpropKernel(self.transformer, op))

    @add_kernel.on_type(TensorSizeOp)
    def add_kernel(self, op):
        self.kernels.append(FlexFillKernel(self.transformer, op.tensor_description(),
                                           op.reduction_axes.size))

    @add_kernel.on_type(update_lut)
    def add_kernel(self, op):
        self.kernels.append(FlexLUTBpropKernel(self.transformer, op))

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
            elif isinstance(kernel, non_flex_kernels):
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
        Flex management before kernel call

        Before a kernel call, flex tensor scales are adjusted
        and new values are bound to kernel params
        """

        # flex management
        self.transformer.flex_manager.manage_before_computation(kernel)

        # bind flex scale kernel parameters
        kernel.bind_flex_scales()

    def after_kernel_execute(self, kernel):
        """
        Flex management after kernel call
        """
        # flex management
        self.transformer.flex_manager.manage_after_computation(kernel)

    def __call__(self):
        self.transformer.flex_manager.autoflex_count += 1

        super(FlexGPUKernelGroup, self).__call__()
