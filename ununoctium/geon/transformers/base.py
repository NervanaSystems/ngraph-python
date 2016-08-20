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


class Computation(with_metaclass(abc.ABCMeta, object)):
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


class DeviceBuffer(with_metaclass(abc.ABCMeta, object)):
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


class DeviceTensor(with_metaclass(abc.ABCMeta, object)):
    """
    A handle to a tensor on the device.

    :ivar device_buffer The device buffer [reference] backing this view.
    :ivar tensor_description: The tensor description for this tensor.
    """
    def __init__(self, transformer, device_buffer, tensor_description, **kwargs):
        super(DeviceTensor, self).__init__(**kwargs)
        self.device_buffer = device_buffer
        self.tensor_description = tensor_description
        device_buffer.views.add(self)

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

    def finalize(self):
        """
        Prepare for allocation.
        """
        Op.simple_prune(self.all_results)

        # Crate tensor descriptions
        ops = Op.ordered_ops(self.all_results)
        self.inits = self.ordered_initializers(ops)
        all_ops = ops + self.inits
        # Give ids
        for op in all_ops:
            if op not in self.opids:
                self.opids[op] = len(self.opids)

        # Create tensor descriptions
        for op in all_ops:
            op.create_tensor_descriptions(self)

        self.dataflow, self.memory = assign_buffers(
            self, self.all_results, self.fusion
        )

        for tensor_description in self.tensor_descriptions:
            tensor_description.initialize()

        self.ops = self.dataflow.instructions
        self.order = {op: i for i, op in enumerate(self.ops)}
        self.initializers = self.ordered_initializers(self.ops)

        # Compile the computations now that we know their storage
        for computation in self.computations:
            ordered_ops = self.dataflow.can_reach(computation.ops, order=self.ops)
            computation.executor = self.compile_computation(ordered_ops)

        self.finalized = True

    def allocate(self):
        """
        Allocate storage.

        Will finalize if not already done.
        """
        if self.allocated:
            return
        if not self.finalized:
            self.finalize()

        # TODO Move to compilation step
        for device_buffer in self.device_buffers:
            device_buffer.allocate()
        for op in self.inits + self.ops:
            self.initialize_constant(op)

        self.allocated = True

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
        self.transform_ordered_ops(self.initializers)
        self.initialized = True

    def compile_computation(self, ordered_ops):
        """
        Return a function that will run the computation in this transformer.

        Should be overridden by transformers.

        Arguments:
          ordered_ops: TODO

        Returns:
          Function that runs the computation
        """
        return lambda: self.transform_ordered_ops(ordered_ops)

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

    def transform_ordered_ops(self, ordered_ops):
        """
        Call op.transform_call_info on every op in ordered_ops.

        If transform_hooks are present on the op or on this transformer, call
        those as well.well

        Arguments:
          ordered_ops: TODO
        """

        def transform_op(op):
            """
            This is the call we would make directly if there were no hooks.
            wrap it up into a function so we can pass it to a hook which has
            the responsibility of making the call to the hook.  This allows the
            hook to execute both before and after the transform.

            Arguments:
              op: TODO
            """
            op.transform_call_info(self)

        for op in ordered_ops:
            if op.transform_hook is not None:
                op.transform_hook(self, op, transform_op)
            elif self.transform_hook is not None:
                self.transform_hook(self, op, transform_op)
            else:
                # run the transform without any hooks
                transform_op(op)

    @abc.abstractmethod
    def device_buffer_storage(self, bytes, alignment):
        """
        Make a DeviceBuffer.

        :param bytes: Size of buffer.
        :param alignment: Alignment of buffer.
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

    @abc.abstractmethod
    def make_raw_buffer(self, size):
        """
        Allocate raw buffer.

        Arguments:
          size: Size in bytes of the buffer to allocate
        """

    @abc.abstractmethod
    def nparray(self, tensor_description, array):
        """
        Allocate a tensor and initialize it with a numpy array.

        This needs to be executed from the CPU since that's where the NumPy array is.

        Arguments:
          tensor_description: TODO
          array: TODO

        Returns:
          Reference to the tensor
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def rng(self, seed=None):
        """
        Allocate a random number generator.

        Arguments:
          seed: An integer.

        Returns:
          Reference to the random number generator.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def tensor_view(self, tensor_description):
        """
        Allocate a view of a tensor.

        Arguments:
          tensor_description: Description of the tensor view.

        Returns:
          Reference to the tensor view.
        """
        raise NotImplementedError()

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

    # Operations
    @abc.abstractmethod
    def absolute(self, x, out):
        """
        Absolute value.

        Arguments:
          x: Input tensor
          out: Output tensor, may be input.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def add(self, x, y, out):
        """
        out = x + y

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def argmax(self, x, out):
        """
        Argmax on dim 0 of x.

        Arguments:
          x: TODO
          out: Integer tensor
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def argmin(self, x, out):
        """
        Argmin on dim 0 of x.

        Arguments:
          x: TODO
          out: Integer tensor
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def cos(self, x, out):
        """
        Cosine.

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def divide(self, x, y, out):
        """
        out = x/y

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def dot(self, x, y, out):
        """
        Generalized dot using NumPy dimension conventions.

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def equal(self, x, y, out):
        """
        Numerical equality.

        Arguments:
          x: TODO
          y: TODO
          out: Boolean tensor.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def exp(self, x, out):
        """
        out = e^x

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def greater(self, x, y, out):
        """
        x > y

        Arguments:
          x: TODO
          y: TODO
          out: Boolean tensor.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def greater_equal(self, x, y, out):
        """
        x >= y

        Arguments:
          x: TODO
          y: TODO
          out: Boolean tensor.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def less(self, x, y, out):
        """
        x < y

        Arguments:
          x: TODO
          y: TODO
          out: Boolean tensor.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def less_equal(self, x, y, out):
        """
        x <= y

        Arguments:
          x: TODO
          y: TODO
          out: Boolean tensor.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def log(self, x, out):
        """
        log(x)

        Arguments:
          x: TODO
          out: Boolean tensor.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def max(self, x, axis, out):
        """
        Maximum x value on axis.

        Arguments:
          x: TODO
          axis: Axis to maximize over.
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def maximum(self, x, y, out):
        """
        max(x, y)

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def min(self, x, axis, out):
        """
        Minimum x value on axis.

        Arguments:
          x: TODO
          axis: Axis to maximize over.
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def minimum(self, x, y, out):
        """
        min(x, y)

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def multiply(self, x, y, out):
        """
        x*y

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def negative(self, x, out):
        """
        -x

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def not_equal(self, x, y, out):
        """
        x != y

        Arguments:
          x: TODO
          y: TODO
          out: Boolean tensor.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def onehot(self, idx, out):
        """
        TODO

        Arguments:
          idx: Index tensor
          out: 2-d tensor, axis 0 gets onehot expansion
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reciprocal(self, x, out):
        """
        1/x

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sign(self, x, out):
        """
        signum(x)

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sin(self, x, out):
        """
        sine(x)

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sqrt(self, x, out):
        """
        sqrt(x)

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def square(self, x, out):
        """
        x^2

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def subtract(self, x, y, out):
        """
        x - y

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sum(self, x, axis, out):
        """
        sum of x over axis

        Arguments:
          x: TODO
          axis: TODO
          out: TODO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def tanh(self, x, out):
        """
        tanh(x)

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    # @abc.abstractmethod
    def allreduce(self, x, out):
        """
        MPI allreduce

        Arguments:
          x: TODO
          out: TODO
        """
        raise NotImplementedError()

    # @abc.abstractmethod
    def conv2d(self, x, y, out):
        """
        2 dimensional convolution

        Arguments:
          x: TODO
          y: TODO
          out: TODO
        """
        raise NotImplementedError()
