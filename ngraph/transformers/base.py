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

import collections
import weakref
import logging

import abc
from builtins import object
from future.utils import with_metaclass

from ngraph.op_graph.op_graph import Op, computation
from ngraph.util.names import NameableValue
from orderedset import OrderedSet

from ngraph.util.trace_events import is_tracing_enabled

PYCUDA_LOGIC_ERROR_CODE = 4

logger = logging.getLogger(__name__)


class UnsupportedTransformerException(Exception):
    pass


class Computation(NameableValue):
    """
    A handle for a computation function.

    Arguments:
        transformer (obj:`Transformer`): The associated transformer.
        returns: If an Op, return the value
            of the Op, if sequence of Ops, return the sequence of values, if
            a set return a map, if None, return None.
        *args: AllocationOps marked input will be arguments to the function.
        **kwargs: Args for related classes.
    """

    def __init__(self, transformer, computation_op, **kwargs):
        super(Computation, self).__init__(**kwargs)
        logging.info("Creating computation with computation_op: %s", computation_op)
        self.transformer = transformer
        self.computation_op = computation_op
        self.computation_name = None
        self.executor = None

        self.send_nodes = []
        self.recv_nodes = []
        self.scatter_send_nodes = []
        self.scatter_recv_nodes = []
        self.gather_send_nodes = []
        self.gather_recv_nodes = []
        self.allreduce_nodes = []
        self.broadcast_send_nodes = []
        self.broadcast_recv_nodes = []

    def unpack_args_or_feed_dict(self, args, kwargs):
        feed_dict = kwargs.pop('feed_dict', None)
        if feed_dict is not None:
            if len(args) != 0:
                raise ValueError((
                    'Can not supply both positional and feed_dict arguments '
                    'to Computation'
                ))

            args = tuple(feed_dict[param.tensor] for param in self.computation_op.parameters)

        if len(args) != len(self.computation_op.parameters):
            raise ValueError((
                'Computation was expecting {expected} arguments, but was '
                'called with {called}.'
            ).format(
                expected=len(self.computation_op.parameters),
                called=len(args),
            ))
        return args

    def __call__(self, *args, **kwargs):
        """
        Executes the computation passing args in to the function.
        """
        args = self.unpack_args_or_feed_dict(args, kwargs)

        # TODO Should this be automatic?
        self.transformer.initialize()

        # Get the parameters to the device
        self.transformer.host_to_device(self, self.computation_op.parameters, args)

        self.executor()

        # Generate a timeline for the computation
        if is_tracing_enabled():
            self.generate_profile(self.executor.__profiler_start__,
                                  self.executor.__profiler_stop__)

        # TODO Should copy this out of the device to a destination when it is not scalar
        def value(op):
            """
            Returns the computed value of op, or None if it has no value.

            :param op:
            :return: Return value for op.
            """
            if op.is_tensor_op:
                return self.transformer.device_to_host(self, op)
            else:
                return None

        if isinstance(self.computation_op.returns, Op):
            return value(self.computation_op.returns)
        elif isinstance(self.computation_op.returns, (collections.Sequence, OrderedSet)):
            return tuple(value(op) for op in self.computation_op.returns)
        elif isinstance(self.computation_op.returns, collections.Set):
            result = dict()
            for op in self.computation_op.returns:
                result[op] = value(op)
            return result
        else:
            return None

    def generate_profile(self, profiler_start, profiler_stop):
        pass


class DeviceBufferStorage(with_metaclass(abc.ABCMeta, NameableValue)):
    """
    Something that can provide storage.

    Arguments:
        transformer: The transformer associated with this device buffer.
        bytes: Size of storage.
        dtype: Alignment of storage.

    Attributes:
        transformer: The transformer associated with this device buffer.
        bytes: Size of storage.
        dtype: Alignment of storage.

        views: All direct tensor views of this buffer.
    """
    def __init__(self, transformer, bytes, dtype, **kwargs):
        super(DeviceBufferStorage, self).__init__(**kwargs)
        self.transformer = transformer
        self.bytes = bytes
        self.dtype = dtype
        self.__views = weakref.WeakValueDictionary()

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

    def device_tensor(self, tensor_description):
        """
        Returns a DeviceTensor for tensor_description.

        Arguments:
            tensor_description: The TensorDescription of the tensor.

        Returns: A DeviceTensor.
        """
        tensor = self.__views.get(tensor_description.parameter_key, None)
        if tensor is not None:
            return tensor
        tensor = self.create_device_tensor(tensor_description)
        self.__views[tensor_description.parameter_key] = tensor
        return tensor

    @abc.abstractmethod
    def create_device_tensor(self, tensor_description):
        """
        Creates a DeviceTensor for tensor_description.

        Arguments:
            tensor_description: The TensorDescription of the tensor.

        Returns: A DeviceTensor.
        """


class DeviceTensor(with_metaclass(abc.ABCMeta, NameableValue)):
    """
    A handle to a tensor on the device.

    Arguments:
        transformer: The associated transformer.
        device_buffer: The device buffer for the elements.
        tensor_description: The tensor_description describing this device tensor.
        **kwargs: Args for related classes.
    """
    def __init__(self, transformer, device_buffer, tensor_description, **kwargs):
        super(DeviceTensor, self).__init__(**kwargs)
        self.transformer = transformer
        self.device_buffer = device_buffer
        self.tensor_description = tensor_description

    @property
    def dtype(self):
        return self.tensor_description.dtype

    @abc.abstractmethod
    def transform_allocate(self):
        """Generate code for making the device tensor usable on the device."""

    @abc.abstractmethod
    def get(self, tensor):
        """
        Copy from device to tensor.

        :param tensor: Destination of copy.  If None, tensor will be allocated.
        :return: tensor.
        """

    @abc.abstractmethod
    def __getitem__(self, key):
        """
        Provides read access to the device tensor.

        :param key: The index/slice
        :return: The item/value.
        """

    @abc.abstractmethod
    def __setitem__(self, key, value):
        """
        Provides write access to the device tensor.

        :param key: The index/slice
        :param value: Tensor with same size as index/slice
        """


class Transformer_ABC_Meta(abc.ABCMeta):
    """
    metaclass for the backend objects
    takes care of registering all the backend subclasses
    """
    def __init__(cls, name, bases, dict_):
        if not hasattr(cls, 'transformers'):
            # First possible transformer class sets things up
            cls.transformers = {}

        # If this transformer has a transformer_name, register it
        transformer_name = getattr(cls, 'transformer_name', None)
        if transformer_name is not None:
            cls.transformers[transformer_name] = cls
        super(Transformer_ABC_Meta, cls).__init__(name, bases, dict_)


class Transformer(with_metaclass(Transformer_ABC_Meta, object)):
    """
    Produce an executable version of op-graphs.

    Computations are subsets of Ops to compute.  The transformer determines storage
    allocation and transforms the computations and allocations into functions.

    Arguments:
        fusion (bool): Whether to combine sequences of operations into one operation.
        **kwargs: Args for related classes.

    Attributes:
        computations (:obj:`set` of :class:`Computation`): The set of requested computations.
        all_results (:obj:`set` of :class:`ngraph.op_graph.op_graph.Op`):  A root set of Ops that
            need to be computed.
        finalized (bool): True when transformation has been performed.
        initialized (bool): True when variables have been initialized/restored.
        fusion (bool): True when fusion was enabled.
        device_buffers (set): Set of handles for storage allocations.
    """
    def __init__(self, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.graph_passes = []
        self.byte_alignment = 4

    @abc.abstractproperty
    def use_exop(self):
        """

        Returns: True if this transformer uses the execution graph.

        """
        pass

    @abc.abstractmethod
    def initialize_allocations(self):
        """
        Inititializes allocation caches.

        """

    @abc.abstractmethod
    def get_tensor_view_value(self, op, host_tensor=None):
        """
        Returns the contents of the tensor view for op.

        Args:
            op: The computation graph op.
            host_tensor: Optional tensor to copy value into.

        Returns:
            A NumPy tensor with the elements associated with op.

        """

    @abc.abstractmethod
    def host_to_device(self, computation, parameters, args):
        """
        Copy args to parameters in computation.

        Args:
            computation: The computation.
            parameters: Parameters of the computation.
            args: Values for the parameters.

        """

    @abc.abstractmethod
    def device_to_host(self, computation, op, tensor=None):
        """
        Copy a computation result from the device back to the host.

        Args:
            computation: The computation.
            op: The op associated with the value.
            tensor: Optional tensor for returned value.

        Returns:
            The value of op.

        """

    def get_layouts(self, op):
        """
        Returns a list of possible axis layouts for the op. The default layout must
        be the first item in the returned list.

        Arguments:
            op: graph op to get possible layouts for

        Returns:
            A list of objects that inherit from LayoutAssignment. The first item in the
            list must be the default layout for this op.
        """
        raise NotImplementedError("Layout methods not implemented in this transformer")

    def get_layout_cost_function(self, op):
        """
        Returns a UnaryLayoutConstraint which computes the cost of an op given an
        assigned data layout for that op.

        Arguments:
            op: graph op to get cost function for

        Returns:
            An object that inherits from UnaryLayoutConstraint and can be used to
            calculate the layout assignment cost.
        """
        raise NotImplementedError("Layout methods not implemented in this transformer")

    def get_layout_change_cost_function(self, op, arg):
        """
        Returns a BinaryLayoutConstraint which computes the cost of a layout change
        between the specified op and its specified arg (if any cost).

        Arguments:
            op: graph op to get cost function for
            arg: argument to the op to generate cost function for

        Returns:
            An object that inherits from BinaryLayoutConstraint and can be used to
            calculate any layout change cost.
        """
        raise NotImplementedError("Layout methods not implemented in this transformer")

    def set_output_statistics_file(self, statistics_file):
        """
        Collects data for transformer
        """
        pass

    def save_output_statistics_file(self):
        """
        Save collected statistics data to file
        """
        pass

    # Old interface
    def computation(self, results, *parameters):
        """
        Adds a computation to the transformer. In the case of not providing parameters
        explicitly, the computation will keep using the old values for the parameters.

        Arguments:
            results: Values to be computed
            *parameters: Values to be set as arguments to evaluate

        Returns:
            Callable.
        """

        return self.add_computation(computation(results, *parameters))

    def add_computation(self, computation):
        return self.make_computation(computation)

    def make_computation(self, computation):
        """
        Wrap in Computation or a transformer-specific subclass.

        Args:
            computation:

        Returns:
            Computation or a subclass.

        """
        return Computation(self, computation)

    def initialize(self):
        """
        Initialize storage.  Will allocate if not already performed.
        """
        pass

    def register_graph_pass(self, graph_pass, position=None):
        """
        Register a graph pass to be run.

        Arguments:
            graph_pass (): The pass to register
            position (int): insert index in the list of passes, append by default
        """
        if position is not None:
            self.graph_passes.insert(position, graph_pass)
        else:
            self.graph_passes.append(graph_pass)

    def close(self):
        pass

    @classmethod
    def get_atol(cls, desired, atol):
        return atol

    def __del__(self):
        self.close()


class ComputationGraphTransformer(Transformer):
    def __init__(self, **kwargs):
        super(ComputationGraphTransformer, self).__init__(**kwargs)
        self.computations = OrderedSet()
        self.device_buffers = None
        self.op_tensors = None
        self.op_tensor_views = None
        self.initialize_allocations()
        self.finalized = False
        self.allocated = False
        self.initialized = False

    def run_registered_graph_passes(self, ops, **kwargs):
        for graph_pass in self.graph_passes:
            graph_pass.wrapped_do_pass(ops=ops, **kwargs)
        return ops

    def initialize_allocations(self):
        """
        Inititializes allocation caches.

        """
        self.op_tensors = dict()
        self.op_tensor_views = dict()
        self.device_buffers = OrderedSet()

    def get_op_tensor(self, op):
        """
        Returns the tensor allocated for this op.

        Args:
            op: A computation graph op.

        Returns:
            A device tensor (DeviceBuffer).

        """
        return self.op_tensors[op.forwarded.tensor.forwarded.tensor_description().base]

    def has_op_tensor(self, op):
        """
        Returns true if the op has a device tensor.

        Args:
            op: A computation graph op.

        Returns:
            True if the op has a device tensor.

        """
        return op.forwarded.tensor.forwarded.tensor_description().base in self.op_tensors

    def get_op_tensor_view(self, op):
        """
        Returns the tensor view for this op.

        Args:
            op: A computation graph op.

        Returns:
            A device tensor view.

        """
        return self.op_tensor_views[op.forwarded.tensor.forwarded.tensor_description()]

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

    def get_tensor_description_tensor(self, tensor_description):
        """
        Returns a tensor for a tensor description.

        Deprecated. Use op version.

        Args:
            tensor_description:

        Returns:
            A device tensor (DeviceBuffer).


        """
        return self.op_tensors[tensor_description.base]

    def has_tensor_description_tensor(self, tensor_description):
        """
        Tests if a tensor description has a device tensor.

        Deprecated, only used by flex.

        Args:
            tensor_description:

        Returns:
            True if the tensor description has a device tensor.

        """
        return tensor_description.base in self.op_tensors

    def get_tensor_description_tensor_view(self, tensor_description):
        """
        Returns a tensor for a tensor description.

        Deprecated, only used by flex.

        Args:
            tensor_description:

        Returns:
            A device tensor view.

        """
        return self.op_tensor_views[tensor_description]

    def host_to_device(self, computation, parameters, args):
        """
        Copy args to parameters in computation.

        Args:
            computation: The computation.
            parameters: Parameters of the computation.
            args: Values for the parameters.

        """
        for param, arg in zip(parameters, args):
            self.get_op_tensor_view(param)[...] = arg

    def device_to_host(self, computation, op, tensor=None):
        """
        Copy a computation result from the device back to the host.

        Args:
            computation: The computation.
            op: The op associated with the value.
            tensor: Optional tensor for returned value.

        Returns:
            The value of op.

        """
        if self.has_op_tensor(op):
            return self.get_op_tensor_view(op).get(tensor)

    @property
    def use_exop(self):
        """

        Returns: True if this transformer uses the execution graph.

        """
        return False

    @abc.abstractmethod
    def start_transform_allocate(self):
        """
        Called just before allocation code is transformed.
        """

    @abc.abstractmethod
    def finish_transform_allocate(self):
        """
        Called after last allocation is transformed.
        """

    @abc.abstractmethod
    def transform_ordered_ops(self, computation, ordered_ops, name):
        """
        Generate code to compute ordered_ops.

        Arguments:
            computation: The computation being compiled.
            ordered_ops: Ops to compute
            name: The name of the computation.

        Returns: Handle for generated code
        """

    @abc.abstractmethod
    def finish_transform(self):
        """
        Finish generating the model.
        """

    @abc.abstractmethod
    def allocate_storage(self):
        """
        Allocate storage on the device.
        """

    @abc.abstractmethod
    def device_buffer_storage(self, bytes, dtype, name):
        """
        Make a DeviceBuffer.

        Arguments:
            bytes: Size of buffer.
            dtype: dtype of buffer.
            name: Name of the storage variable

        returns: A DeviceBuffer.
        """

    def allocate(self):
        """
        Allocate storage and then initializes constants.

        Will finalize if not already done.
        """
        if self.allocated:
            return

        if not self.finalized:
            logging.info("Finalizing transformer.")
            self._transform_computations()

        self.allocate_storage()

        init_states = OrderedSet()
        for op in OrderedSet(self.ops):
            op = op.forwarded
            states = op.states_read | op.states_written
            for state in states:
                state = state.forwarded
                if state.initial_value is not None:
                    init_states.add(state)
        for state in init_states:
            tensor_description = state.tensor.tensor_description()
            self.get_tensor_description_tensor_view(tensor_description)[...] = state.initial_value

        self.allocated = True

    def add_computation(self, computation):
        """
        Adds a computation to the transformer.

        Arguments:
            computation: A computation Op.

        Returns:
            Callable.
        """
        if self.finalized:
            raise ValueError(
                'Cannot create computations from a finalized transformer'
            )
        computation = super(ComputationGraphTransformer, self).add_computation(computation)
        self.computations.add(computation)
        return computation

    def initialize(self):
        """
        Initialize storage.  Will allocate if not already performed.
        """
        if self.initialized:
            return
        self.allocate()

        # Need to set initialized before we are done because the init computation will
        # try to initialize.
        self.initialized = True

    def _transform_computations(self):
        """
        Transform computation graphs to a form that can be run.
        """

        # Run passes on the computation graphs
        all_results = []
        for comp in self.computations:
            all_results.append(comp.computation_op)

        all_ops = self.run_registered_graph_passes(ops=all_results)

        # Collect up all ops from the graph and obtain the init graph
        all_ops = OrderedSet(Op.ordered_ops(all_ops))

        def ensure_tensor(op):
            op = op.forwarded
            tensor_description = op.tensor_description()
            base = tensor_description.base
            tensor = self.op_tensors.get(base, None)
            if tensor is None:
                tensor = self.device_buffer_storage(
                    base.tensor_size,
                    base.dtype,
                    base.name
                )
                self.op_tensors[base] = tensor
                self.device_buffers.add(tensor)
            tensor_view = tensor.device_tensor(tensor_description)
            self.op_tensor_views[tensor_description] = tensor_view

        self.ops = Op.ordered_ops(all_ops)
        for op in self.ops:
            if op.is_tensor_op:
                ensure_tensor(op)

        self.start_transform_allocate()
        for device_buffer in self.device_buffers:
            device_buffer.transform_allocate()
        self.finish_transform_allocate()

        # Compile the computations now that we know their storage
        for comp in self.computations:
            comp.computation_name = \
                self.transform_ordered_ops(comp,
                                           Op.ordered_ops([comp.computation_op]),
                                           name=comp.name)
        self.finish_transform()
        self.finalized = True


__transformer_factory = None


def make_transformer():
    """
    Generates a Transformer using the factory in this module which defaults
    to CPU

    Returns: Transformer
    """
    return __transformer_factory()


def set_transformer_factory(factory):
    """
    Sets the Transformer factory used by make_transformer

    Arguments:
        factory (object): Callable object which generates a Transformer
    """
    global __transformer_factory
    __transformer_factory = factory


def transformer_choices():
    """Return the list of available transformers."""
    names = sorted(Transformer.transformers.keys())
    return names


def allocate_transformer(name, **kargs):
    """Allocate a named backend."""
    try:
        return Transformer.transformers[name](**kargs)
    except KeyError:
        names = ', '.join(["'%s'" % (_,) for _ in transformer_choices()])
        raise ValueError("transformer must be one of (%s)" % (names,))


def make_transformer_factory(name, **kargs):
    def factory():
        return allocate_transformer(name, **kargs)
    factory.name = name  # added for pytest
    return factory
