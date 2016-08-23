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

import abc
from builtins import object
import numbers
import collections
from future.utils import with_metaclass

import numpy as np

from geon.backends.graph.environment import get_current_environment
from geon.op_graph.op_graph import Op, placeholder, TensorOp, InitTensor, tensor_descriptions
from geon.analysis.memory import assign_buffers
from geon.util.generics import generic_method
from geon.op_graph.names import NameableValue


class Computation(with_metaclass(abc.ABCMeta, NameableValue)):
    def __init__(self, transformer, returns, *args):
        """
        Defines computation.

        Arguments:
          transformer: TODO
          returns: If an Op, return the value of the Op, if sequence of Ops, return
                   the sequence of values, if a Set return a map, if None, return None.
          args: Placeholders will be arguments to the function, other values are ops
                to compute but not return.
        """
        self.transformer = transformer
        self.returns = returns
        self.ops = set()
        if isinstance(returns, collections.Set):
            self.ops.update(returns)
        elif isinstance(returns, collections.Sequence):
            self.ops.update(returns)
        elif isinstance(returns, Op):
            self.ops.add(returns)
        elif returns is None:
            pass
        else:
            raise ValueError()

        self.parameters = []
        for arg in args:
            if isinstance(arg, placeholder):
                self.parameters.append(arg)
            if isinstance(arg, Op):
                self.ops.add(arg)
            else:
                raise ValueError()

        self.transformer.all_results.update(self.ops)
        self.executor = None
        self.name = None

    def transform(self):
        ordered_ops = self.transformer.dataflow.can_reach(self.ops, order=self.transformer.ops)
        self.name = self.transformer.transform_ordered_ops(ordered_ops)

    def __call__(self, *args):
        # TODO Should this be automatic?
        self.transformer.initialize()

        # Get the parameters to the device
        for param, arg in zip(self.parameters, args):
            self.transformer.copy_to_model(param, arg)
        self.executor()

        # TODO Should copy this out of the device to a destination when it is not scalar
        def value(op):
            """
            Returns the computed value of op, or None if it has no value.

            :param op:
            :return: Return value for op.
            """
            if isinstance(op, TensorOp):
                return op.tensor_description(self.transformer).value.get(None)
            else:
                return None

        if isinstance(self.returns, Op):
            return value(self.returns)
        elif isinstance(self.returns, collections.Set):
            result = dict()
            for op in self.returns:
                dict[op] = value(op)
            return result

        elif isinstance(self.returns, collections.Sequence):
            return tuple(value(op) for op in self.returns)

        else:
            return None


class DeviceBuffer(with_metaclass(abc.ABCMeta, NameableValue)):
    """
    Something that can provide storage.

    :ivar transformer: The transformer associated with this device buffer.
    :ivar views: All direct tensor views of this buffer.
    """
    def __init__(self, transformer, **kwargs):
        """

        :param transformer: The associated transformer.
        :param kwargs: Any additional arguments.
        """
        super(DeviceBuffer, self).__init__(**kwargs)
        self.transformer = transformer
        self.views = set()
        self.transformer.device_buffers.add(self)

    def generate_allocate(self):
        """
        Generate allocation code.
        """
        self.generate_allocate_views()

    def generate_allocate_views(self):
        """Generate code for allocating views"""
        for view in self.views:
            view.generate_allocate()

    def allocate(self):
        """
        Allocate storage on the device.

        Finish by allocating views views.
        """
        self.allocate_views()

    def allocate_views(self):
        """
        Allocate all views of this buffer.
        """
        for view in self.views:
            view.allocate()

    @property
    @abc.abstractmethod
    def storage_device_buffer(self):
        """
        Get the actual storage buffer.

        :return: A DeviceBufferStorage
        """
        raise NotImplementedError()


class DeviceBufferStorage(with_metaclass(abc.ABCMeta, DeviceBuffer)):
    """
    A handle to device storage.

    :ivar bytes: The size of the byte buffer.
    :ivar alignment: The alignment of the byte buffer.
    """
    def __init__(self, transformer, bytes, alignment, **kwargs):
        """

        :param transformer: The associated transformer.
        :param bytes: Size of storage.
        :param alignment: Alignment of storage.
        :param kwargs: Additional args.
        """
        super(DeviceBufferStorage, self).__init__(transformer, **kwargs)
        self.bytes = bytes
        self.alignment = alignment

    @property
    def storage_device_buffer(self):
        return self


class DeviceBufferReference(with_metaclass(abc.ABCMeta, DeviceBuffer)):
    """
    Holds a reference to a DeviceBuffer.

    :ivar transformer: The transformer associated with this device buffer reference.
    :ivar views: All direct tensor views of this buffer reference.  Does not include views of
     device buffer references set to this device buffer.
    """
    def __init__(self, transformer, **kwargs):
        super(DeviceBufferReference, self).__init__(self, **kwargs)
        self.__device_buffer = None

    @property
    def storage_device_buffer(self):
        return self.device_buffer.storage_device_buffer

    @property
    def device_buffer(self):
        """
        The referenced device buffer.
        """
        return self.__device_buffer

    @device_buffer.setter
    def device_buffer(self, device_buffer):
        """
        Change the referenced device buffer.

        :param device_buffer: The new device buffer.
        """
        self.__device_buffer = device_buffer
        self.allocate_views()


class DeviceTensor(with_metaclass(abc.ABCMeta, NameableValue)):
    """
    A handle to a tensor on the device.

    :ivar device_buffer The device buffer [reference] backing this view.
    :ivar tensor_description: The tensor description for this tensor.
    """
    def __init__(self, transformer, device_buffer, tensor_description, **kwargs):
        super(DeviceTensor, self).__init__(**kwargs)
        self.transformer = transformer
        self.device_buffer = device_buffer
        self.tensor_description = tensor_description
        device_buffer.views.add(self)

    @property
    def dtype(self):
        return self.tensor_description.dtype

    @abc.abstractmethod
    def generate_allocate(self):
        """Generate code for making the device tensor usable on the device."""

    @abc.abstractmethod
    def allocate(self):
        """
        Make the device tensor usable on the device.
        """

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


class Transformer(with_metaclass(abc.ABCMeta, object)):
    """
    Given a list of ops you want to compute the results of, this transformer
    will compile the graph required to compute those results and exposes an
    evaluate method to execute the compiled graph.
    """

    def __init__(self, environment=None, fusion=None, **kvargs):
        """
        TODO.

        Arguments:
          environment: The environment to use to grab things like axis.  WARNING: `environment`
                       will be deprecated soon.
          fusion: Whether to combine sequences of operations into one operation.
        """
        super(Transformer, self).__init__(**kvargs)
        self.transform_hook = None
        if environment is None:
            environment = get_current_environment()
        self.environment = environment
        self.computations = set()
        self.all_results = set()
        self.values = dict()
        self.cache = dict()
        self.tensor_descriptions = set()
        self.finalized = False
        self.allocated = False
        self.initialized = False
        self.opids = dict()
        self.fusion = fusion
        self.device_buffers = set()
        self.cpu_initializations = []
        self.init_computation = None

    def transform_computations(self):
        """
        Transform computation graphs to a form that can be run.
        """
        Op.simple_prune(self.all_results)

        # Crate tensor descriptions
        ops = Op.ordered_ops(self.all_results)
        self.inits = self.ordered_initializers(ops)

        self.init_computation = self.computation([], *self.inits)

        all_ops = ops + self.inits
        # Give ids
        for op in all_ops:
            if op not in self.opids:
                self.opids[op] = len(self.opids)

        # Create tensor descriptions
        for op in all_ops:
            op.create_tensor_descriptions(self)

        self.dataflow, self.memory = assign_buffers(
            self, self.all_results.union(self.inits), self.fusion
        )

        for tensor_description in self.tensor_descriptions:
            tensor_description.initialize()

        self.ops = self.dataflow.instructions
        self.order = {op: i for i, op in enumerate(self.ops)}
        self.initializers = self.ordered_initializers(self.ops)

        self.start_transfrom_allocate()
        for device_buffer in self.device_buffers:
            device_buffer.generate_allocate()
        self.finish_transfrom_allocate()

        # Compile the computations now that we know their storage
        for computation in self.computations:
            computation.transform()
        self.init_computation.transform()
        self.generate_model()
        self.finalized = True

    @abc.abstractmethod
    def start_transfrom_allocate(self):
        """
        Called just before allocation code is generated.
        """

    @abc.abstractmethod
    def finish_transfrom_allocate(self):
        """
        Called after last allocation is generated.
        """

    @abc.abstractmethod
    def transform_ordered_ops(self, ordered_ops):
        """
        Generate code to compute ordered_ops.

        :param ordered_ops: Ops to compute
        :return: Handle for generated code
        """

    @abc.abstractmethod
    def generate_model(self):
        """
        Finish generating the model.
        """

    def allocate(self):
        """
        Allocate storage.

        Will finalize if not already done.
        """
        if self.allocated:
            return
        if not self.finalized:
            self.transform_computations()

        self.allocate_storage()

        for op in self.inits + self.ops:
            self.initialize_constant(op)

        self.allocated = True

    @abc.abstractmethod
    def allocate_storage(self):
        """
        Allocate storage on the device.
        """

    @generic_method
    def initialize_constant(self, op):
        pass

    @initialize_constant.on_type(InitTensor)
    def initialize_constant(self, op):
        tensor_description, = tensor_descriptions(op.args, self)
        value = op.valfun(tensor_description)
        tensor_description.value[:] = value

    def initialize(self):
        """
        Initialize storage.  Will allocate if not already performed.
        """
        if self.initialized:
            return
        self.allocate()
        # self.transform_ordered_ops(self.initializers)
        self.initialized = True
        self.init_computation()

    def computation(self, results, *parameters):
        """
        Adds a computation to the transformer.

        Arguments:
          results: Values to be computed
          parameters: Values to be set as arguments to evaluate

        Returns:
          Dictionary from results to their values
        """
        if self.finalized:
            raise ValueError(
                'Cannot create computations from a finalized transformer'
            )

        result = Computation(self, results, *parameters)
        self.computations.add(result)
        return result

    def copy_to_model(self, tensor_op, value):
        """
        TODO.

        Arguments:
          tensor_op: TODO
          value: TODO

        Returns:

        """
        self.allocate()
        td = tensor_op.tensor_description(self)
        if isinstance(value, numbers.Real):
            self.fill(td.value, value)
        elif isinstance(value, np.ndarray):
            if td.full_sizes != value.shape:
                raise ValueError()
            self.set_item(td.value, (), value)
        else:
            raise ValueError()

    def ordered_initializers(self, ordered_ops):
        """
        TODO.

        Arguments:
          ordered_ops: TODO

        Returns:

        """
        todo = set(ordered_ops)
        initializers = set()
        while todo:
            these_ops = todo
            todo = set()
            for op in these_ops:
                initializers.update(op.initializers)
                todo.update(op.initializers)

        ordered_initializer_ops = []
        visited = set()
        inits = set()

        def visit(node):
            """
            TODO.

            Arguments:
              node: TODO

            Returns:

            """
            if node not in visited:
                if node.initializers:
                    if node in inits:
                        if node not in visited:
                            ordered_initializer_ops.append(node)
                            visited.add(node)
                    else:
                        inits.add(node)
                        for n in node.initializers:
                            visit(n)
                else:
                    for n in node.args:
                        visit(n)
                if node not in visited:
                    ordered_initializer_ops.append(node)
                    visited.add(node)

        for node in initializers:
            visit(node)

        return ordered_initializer_ops

    @abc.abstractmethod
    def device_buffer_storage(self, bytes, alignment, name):
        """
        Make a DeviceBuffer.

        :param bytes: Size of buffer.
        :param alignment: Alignment of buffer.
        :param name: Name of the storage variable
        :return: A DeviceBuffer.
        """

    @abc.abstractmethod
    def device_buffer_reference(self):
        """
        Make a DeviceBufferReference.

        :return: A DeviceBufferReference.
        """

    @abc.abstractmethod
    def device_tensor(self, tensor_description):
        """
        Make a DeviceTensor.

        :param tensor_description: The TensorDescription of the tensor.
        :return: A DeviceTensor.
        """

    # Side-effects
    # TODO Should this be combined with set_item?
    @abc.abstractmethod
    def fill(self, out, value):
        """
        Initialize a tensor with a scalar.

        Arguments:
          out: Tensor to initialize
          value: Scalar value.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_item(self, tensor, item, value):
        """
        Implements __setitem__.

        Arguments:
          tensor: Tensor to be modified
          item: Slice/index to set
          value: New values for tensor[item]
        """
        raise NotImplementedError()
