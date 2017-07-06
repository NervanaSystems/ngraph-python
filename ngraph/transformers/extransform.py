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
from __future__ import division
from future.utils import iteritems, itervalues
import abc
from future.utils import with_metaclass
import weakref

from ngraph.util.names import NameableValue
from ngraph.op_graph.op_graph import AssignableTensorOp, TensorValueOp
from ngraph.transformers.base import Transformer
from ngraph.transformers.base import DeviceTensor as BaseDeviceTensorView
from ngraph.transformers.base import Computation as BaseDeviceComputation
from ngraph.transformers.exop import ExecutionState
from ngraph.transformers.passes.exopdelegate import ExOpGraphOpAccessor


class DeviceComputation(BaseDeviceComputation):
    """
    A callable that can run computations on the device.
    """
    def __init__(self, transformer, computation_op, **kwargs):
        super(DeviceComputation, self).__init__(transformer, computation_op, **kwargs)


class DeviceBuffer(NameableValue):
    def __init__(self, transformer, buffer, **kwargs):
        super(DeviceBuffer, self).__init__(name=buffer.buffer_name, **kwargs)
        self.transformer = transformer
        self.device_computation = transformer.device_computation
        self.size = buffer.size
        self.device_tensors = dict()

    def device_tensor(self, tensor_decl, offset=0):
        """
        Get a device tensor based at offset.

        Args:
            tensor_decl: A TensorDecl.
            offset: Byte offset to this buffer.

        Returns:
            The device tensor.

        """
        if tensor_decl.is_compile_only:
            raise ValueError("Allocating compile-only tensor")
        device_tensor = self.device_tensors.get(tensor_decl.buffer_key, None)
        if device_tensor is None:
            device_tensor = self.transformer.make_device_tensor(self.device_computation,
                                                                tensor_decl)
            self.device_tensors[tensor_decl.buffer_key] = device_tensor
        return device_tensor

    def codegen(self):
        pass


class DeviceTensor(with_metaclass(abc.ABCMeta, object)):
    """
    Something that can provide storage.

    Arguments:
        transformer: The transformer associated with this device buffer.
        computation: The computation.
        tensor_decl: The associated TensorDecl.

    Attributes:
        transformer: The transformer associated with this device buffer.
        bytes: Size of storage.
        dtype: Alignment of storage.

        views: All direct tensor views of this buffer.
    """

    def __init__(self, transformer, device_computation, tensor_decl, **kwargs):
        super(DeviceTensor, self).__init__(**kwargs)
        self.transformer = transformer
        self.name = tensor_decl.variable_name
        if tensor_decl.is_compile_only:
            raise ValueError("Storage allocation for compile-only tensor")
        self.tensor_decl = tensor_decl
        self.device_computation = device_computation
        self.__views = weakref.WeakValueDictionary()

    @property
    def is_persistent(self):
        return self.tensor_decl.is_persistent

    @property
    def size(self):
        return self.tensor_decl.size

    @property
    def buffer_pool_offset(self):
        return self.tensor_decl.buffer_pool_offset

    @property
    def element_type(self):
        return self.tensor_decl.element_type

    @property
    def views(self):
        """

        Returns: Iterator over views of the buffer

        """
        return self.__views.values()

    def transform_allocate(self):
        """
        Generate allocation code.
        """
        self.transform_allocate_views()

    def transform_allocate_views(self):
        """Generate code for allocating views"""
        for view in self.views:
            view.transform_allocate()

    def device_tensor_view(self, tensor_view_decl):
        """
        Returns a DeviceTensor for tensor_view_decl.

        Arguments:
            tensor_view_decl: The view of the tensor.

        Returns: A DeviceTensor.
        """
        tensor_view = self.__views.get(tensor_view_decl.key, None)
        if tensor_view is not None:
            return tensor_view
        tensor_view = self.make_device_tensor_view(tensor_view_decl)
        self.__views[tensor_view_decl.key] = tensor_view
        return tensor_view

    @abc.abstractmethod
    def make_device_tensor_view(self, tensor_view_decl):
        """
        Creates a DeviceTensorView for tensor_view_decl.

        Arguments:
            tensor_view_decl: The view of the tensor.

        Returns: A DeviceTensorView.
        """


class DeviceTensorView(BaseDeviceTensorView):
    """
    Extends DeviceBuffer with exop behavior.

    Arguments:
        device_tensor:
            The device tensor for associated with this view.
        tensor_view_decl:
            The description of the tensor view to create.
    """
    def __init__(self, device_tensor, tensor_view_decl, **kwargs):
        super(DeviceTensorView, self).__init__(device_tensor.transformer,
                                               device_tensor,
                                               tensor_view_decl.tensor_description,
                                               **kwargs)
        self.name = tensor_view_decl.name
        self.device_tensor = device_tensor
        self.tensor_view_decl = tensor_view_decl

    def transform_allocate(self):
        raise ValueError("Deprecated API")


class ExecutionGraphTransformer(Transformer):
    def __init__(self, **kwargs):
        super(ExecutionGraphTransformer, self).__init__(**kwargs)
        self.execution_state = ExecutionState(self)
        self.device_buffers = dict()
        self.device_tensors = dict()
        self.device_tensor_views = dict()
        self.device_computations = dict()
        self.device_initializations = dict()

    @property
    def use_exop(self):
        """

        Returns: True if this transformer uses the execution graph.

        """
        return True

    def run_registered_graph_passes(self, computation_decl, **kwargs):
        op_accessor = ExOpGraphOpAccessor()
        for graph_pass in self.graph_passes:
            graph_pass.wrapped_do_pass(op_accessor=op_accessor,
                                       computation_decl=computation_decl,
                                       **kwargs)

    @abc.abstractmethod
    def make_device_tensor(self, computation, tensor_decl):
        """
        Make a DeviceTensor.

        Arguments:
            computation:
            tensor_decl: An TensorDecl.

        returns: A DeviceTensor.
        """

    def initialize_allocations(self):
        """
        Inititializes allocation caches.

        """
        raise ValueError()

    def get_op_tensor_view(self, op):
        """
        Returns the tensor view for this op.

        Args:
            op: A computation graph op.

        Returns:
            A device tensor view.

        """
        if isinstance(op, AssignableTensorOp):
            tensor_decl = self.execution_state.get_op_tensor(op)
            return self.device_tensor_view(tensor_decl.values[0].tensor_view_decl)
        else:
            raise ValueError()

    def get_tensor_view_value(self, op, host_tensor=None):
        """
        Returns the contents of the tensor view for op.

        Args:
            op: The computation graph op.
            host_tensor: Optional tensor to copy value into.

        Returns:
            A NumPy tensor with the elements associated with op.

        """
        return self.get_op_tensor_view(op).get(host_tensor)

    def load_computation(self, computation_decl):
        """
        Load a computation and associated storage into the current execution state.

        Args:
            computation_decl: A ComputationDecl for the computation.

        Returns:
            An executable for the computation.

        """
        self.device_computation = computation_decl.device_computation
        exop_block = computation_decl.exop_block
        self.start_allocate_computation(computation_decl)
        for input_decl in itervalues(computation_decl.op_returns):
            self.device_tensor_view(input_decl.tensor_view_decl)
        for exop in exop_block:
            for input_decl in exop.input_decls:
                self.device_tensor_view(input_decl.tensor_view_decl)
            for input_decl in exop.write_args:
                self.device_tensor_view(input_decl.tensor_view_decl)
            for output_decl in exop.output_decls:
                self.device_tensor_view(output_decl.tensor_view_decl)
        # Make sure we have values for ops that got optimized out
        for input_decl in computation_decl.returns.input_decls:
            output_decl = input_decl.source_output_decl
            if isinstance(output_decl.exop.op, TensorValueOp):
                tensor_decl = exop.computation_graph.get_tensor_decl(
                    op=output_decl.exop.op.value_tensor)
                self.device_tensor_view(
                    tensor_decl.get_tensor_view(output_decl.exop.op.tensor_description()))
            else:
                self.device_tensor_view(output_decl.tensor_view_decl)
        for param in computation_decl.computation_op.parameters:
            tensor_decl = computation_decl.get_tensor_decl(op=param.tensor)
            self.device_tensor_view(tensor_decl.root_tensor_view_decl)
        self.finish_allocate_computation(computation_decl)
        self.start_define_computation(computation_decl)
        for exop in exop_block:
            self.generate_exop(exop)
        self.finish_define_computation(computation_decl)
        executor = self.finish_load_computation(computation_decl)
        self.run_device_tensor_initializations()
        return executor

    def start_allocate_computation(self, computation):
        pass

    def finish_allocate_computation(self, computation):
        pass

    def start_define_computation(self, computation):
        pass

    def finish_define_computation(self, computation):
        pass

    def finish_load_computation(self, computation):
        pass

    def make_device_buffer(self, buffer):
        return DeviceBuffer(self, buffer)

    def device_buffer(self, exop_buffer):
        """
        Return the storage associated with buffer, creating if necessary.

        Args:
            exop_buffer:

        Returns:

        """
        device_buffer = self.device_buffers.get(exop_buffer, None)
        if device_buffer is None:
            device_buffer = self.make_device_buffer(exop_buffer)
            self.device_buffers[exop_buffer] = device_buffer
            device_buffer.codegen()

        return device_buffer

    def device_tensor_from_tensor_decl(self, tensor_decl):
        """
        Returns the device tensor, creating if necessary.

        Args:
            tensor_decl:

        Returns:

        """
        device_tensor = self.device_tensors.get(tensor_decl, None)
        if device_tensor is None:
            # buffer = tensor.buffer
            buffer = tensor_decl
            device_buffer = self.device_buffer(buffer)
            device_tensor = device_buffer.device_tensor(tensor_decl)
            self.device_tensors[tensor_decl] = device_tensor
            device_tensor.codegen()
        return device_tensor

    def device_tensor_view(self, tensor_view_decl):
        """
        Returns the device_tensor, creating if necessary.

        Args:
            tensor_view_decl: The tensor view.

        Returns:

        """
        if tensor_view_decl.tensor_decl.is_compile_only:
            return None
        device_tensor_view = self.device_tensor_views.get(tensor_view_decl, None)
        if device_tensor_view is None:
            tensor_decl = tensor_view_decl.tensor_decl
            device_tensor = self.device_tensor_from_tensor_decl(tensor_decl)
            device_tensor_view = device_tensor.device_tensor_view(tensor_view_decl)
            self.device_tensor_views[tensor_view_decl] = device_tensor_view
            device_tensor_view.codegen()
            if tensor_decl.initial_value is not None \
                    or tensor_decl.is_persistent \
                    or tensor_decl.is_input:
                init_device_tensor_view = self.device_tensor_view(
                    tensor_decl.root_tensor_view_decl)
                if tensor_decl.initial_value is not None:
                    self.add_device_tensor_initialization(init_device_tensor_view,
                                                          tensor_decl.initial_value)
        return device_tensor_view

    def add_device_tensor_initialization(self, device_tensor_view, host_tensor):
        self.device_initializations[device_tensor_view] = host_tensor

    def run_device_tensor_initializations(self):
        for device_tensor_view, host_tensor in iteritems(self.device_initializations):
            device_tensor_view[()] = host_tensor
        self.device_initializations = dict()

    def host_to_device(self, device_computation, parameters, args):
        computation_decl = device_computation.computation_decl
        for op, arg in zip(parameters, args):
            tensor_decl = computation_decl.get_tensor_decl(op=op.tensor)
            device_tensor = self.device_tensor_view(tensor_decl.root_tensor_view_decl)
            device_tensor[()] = arg

    def device_to_host(self, device_computation, op, tensor=None):
        computation_decl = device_computation.computation_decl
        if isinstance(op, AssignableTensorOp):
            tensor_decl = computation_decl.get_tensor_decl(op=op)
            device_tensor = self.device_tensor_view(tensor_decl.root_tensor_view_decl)
        else:
            tensor_view = computation_decl.op_returns[op.tensor].tensor_view_decl
            device_tensor = self.device_tensor_view(tensor_view)

        return device_tensor.get(tensor)

    computation_count = 0

    def add_computation(self, computation_op):
        """
        Adds a computation to the transformer.

        Arguments:
            computation_op: A computation Op.

        Returns:
            Callable.

        """
        device_computation = self.device_computations.get(computation_op, None)
        if device_computation is not None:
            return device_computation

        execution_graph = self.execution_state.make_execution_graph(computation_op)
        computation_decl = execution_graph.computation_decl
        self.run_registered_graph_passes(computation_decl=computation_decl)
        ExecutionGraphTransformer.computation_count += 1

        device_computation = self.make_computation(computation_op)
        computation_decl.device_computation = device_computation
        device_computation.computation_decl = computation_decl
        self.device_computations[computation_op] = device_computation

        device_computation.executor = self.load_computation(computation_decl)

        return device_computation
