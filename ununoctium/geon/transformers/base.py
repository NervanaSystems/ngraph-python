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
from future.utils import with_metaclass

from geon.backends.graph.environment import get_current_environment
from geon.op_graph.op_graph import Op
from geon.analysis.memory import assign_buffers


class Computation(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, transformer, results):
        self.transformer = transformer
        self.results = results

    def evaluate(self):
        return self.transformer.evaluate(self.results)


class Transformer(with_metaclass(abc.ABCMeta, object)):
    """
    given a list of ops you want to compute the results of, this transformer
    will compile the graph required to compute those results and exposes an
    evaluate method to execute the compiled graph.
    """

    def __init__(self, results=None, environment=None, **kvargs):
        """
        :param results: a list of Ops whose results the Transformer should
            return on `.evaluate()`.  There aren't any good reasons to initialize a
            transformer with None except for in tests.
        :param environment: the environment to use to grab things like axis.
            WARNING: `environment` will be depricated soon.
        """
        super(Transformer, self).__init__(**kvargs)
        self.transform_hook = None
        if environment is None:
            environment = get_current_environment()
        self.environment = environment
        self.all_results = set()
        self.values = dict()
        self.cache = dict()
        self.tds = set()
        self.finalized = False
        self.opids = dict()

        if results is not None:
            self.all_results.update(results)
            self.finalize()

    def finalize(self):
        Op.simple_prune(self.all_results)
        self.dataflow, self.memory = assign_buffers(self, self.all_results)
        self.ops = self.dataflow.instructions
        self.order = {op: i for i, op in enumerate(self.ops)}
        self.initializers = self.ordered_initializers(self.ops)
        self.initialize_views(self.initializers)
        self.initialize_views(self.ops)
        self.initialize_tds()
        self.allocate_ordered_ops(self.initializers)
        self.allocate_ordered_ops(self.ops)
        self.transform_ordered_ops(self.initializers)
        self.finalized = True

    def computation(self, results):
        if self.finalized:
            raise ValueError(
                'Cannot create computations from a finalized transformer'
            )
        self.all_results.update(results)
        return Computation(self, results)

    def initialize_views(self, ordered_ops):
        # Give ids
        for op in ordered_ops:
            if op not in self.opids:
                self.opids[op] = len(self.opids)

        # Create tensor descriptions
        for op in ordered_ops:
            op.create_tds(self)

    def initialize_tds(self):
        for td in self.tds:
            td.initialize()

    def ordered_initializers(self, ordered_ops):
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

    def allocate_ordered_ops(self, ordered_ops):
        # Allocate
        for op in ordered_ops:
            op.allocate(self)

    def transform_ordered_ops(self, ordered_ops):
        """
        call op.transform_call_info on every op in ordered_ops.

        if transform_hooks are present on the op or on this transformer, call
        those as well.well
        """
        # ???
        for op in ordered_ops:
            op.sync(self)

        def transform_op(op):
            """
            this is the call we would make directly if there were no hooks.
            wrap it up into a function so we can pass it to a hook which has
            the responsibility of making the call to the hook.  This allows the
            hook to execute both before and after the transform.
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

    def evaluate(self, results=None):
        """
        evaluate the compiled graph (stored in self) and return a dictionary of
        results.

        The dictionary will be of the form: {
            Op: value,
            ...
        }
        """
        if results is None:
            results = self.all_results
        ordered_ops = sorted(Op.ordered_ops(results), key=self.order.get)
        self.transform_ordered_ops(ordered_ops)

        return {op: op.output_value(self) for op in results}

    def set_value(self, op, tensor):
        op.tensor_description(self).value = tensor

    @abc.abstractmethod
    def make_raw_buffer(self, size):
        """
        Allocate raw buffer

        :param size: Size in bytes of the buffer to allocate
        """

    @abc.abstractmethod
    def nparray(self, tensor_description, array):
        """
        Allocate a tensor and initialize it with a numpy array.

        This needs to be executed from the CPU since that's where the NumPy array is.

        :param tensor_description:
        :param array:
        :return: Reference to the tensor
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def rng(self, seed=None):
        """
        Allocate a random number generator.

        :param seed: An integer.
        :return: Reference to the random number generator.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def rng_uniform_tensor(self, rng, tensor_description, low, high):
        """
        Allocate a tensor initialized with a uniform distribution.

        :param rng: Random number generator
        :param tensor_description: Description of the tensor's type, shape, size, and strides.
        :param low:
        :param high:
        :return: Reference to uniform distribution.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def rng_normal_tensor(self, rng, tensor_description, loc, scale):
        """
        Allocate a tensor initialized with a uniform distribution.

        :param rng: Random number generator
        :param tensor_description: Description of the tensor's type, shape, size, and strides.
        :param loc:
        :param scale:
        :return: Reference to normal distribution.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def tensor_view(self, tensor_description):
        """
        Allocate a view of a tensor.

        :param tensor_description: Description of the tensor view.
        :return: Reference to the tensor view.
        """
        raise NotImplementedError()

    # Side-effects
    # TODO Should this be combined with set_item?
    @abc.abstractmethod
    def fill(self, out, value):
        """
        Initialize a tensor with a scalar.

        :param out: Tensor to initialize
        :param value: Scalar value.
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_item(self, tensor, item, value):
        """
        Implements __setitem__.

        :param tensor: Tensor to be modified
        :param item: Slice/index to set
        :param value: New values for tensor[item]
        :return:
        """
        raise NotImplementedError()

    # Operations
    @abc.abstractmethod
    def absolute(self, x, out):
        """
        Absolute value.

        :param x: Input tensor
        :param out: Output tensor, may be input.
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def add(self, x, y, out):
        """
        out = x + y

        :param x:
        :param y:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def argmax(self, x, out):
        """
        Argmax on dim 0 of x.

        :param x:
        :param out: Integer tensor
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def argmin(self, x, out):
        """
        Argmin on dim 0 of x.

        :param x:
        :param out: Integer tensor
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def cos(self, x, out):
        """
        Cosine.

        :param x:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def divide(self, x, y, out):
        """
        out = x/y

        :param x:
        :param y:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def dot(self, x, y, out):
        """
        Generalized dot using NumPy dimension conventions.

        :param x:
        :param y:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def equal(self, x, y, out):
        """
        Numerical equality.

        :param x:
        :param y:
        :param out: Boolean tensor.
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def exp(self, x, out):
        """
        out = e^x

        :param x:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def greater(self, x, y, out):
        """
        x > y

        :param x:
        :param y:
        :param out: Boolean tensor.
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def greater_equal(self, x, y, out):
        """
        x >= y

        :param x:
        :param y:
        :param out: Boolean tensor.
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def less(self, x, y, out):
        """
        x < y

        :param x:
        :param y:
        :param out: Boolean tensor.
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def less_equal(self, x, y, out):
        """
        x <= y

        :param x:
        :param y:
        :param out: Boolean tensor.
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def log(self, x, out):
        """
        log(x)

        :param x:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def max(self, x, axis, out):
        """
        Maximum x value on axis.

        :param x:
        :param axis: Axis to maximize over.
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def maximum(self, x, y, out):
        """
        max(x, y)

        :param x:
        :param y:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def min(self, x, axis, out):
        """
        Minimum x value on axis.

        :param x:
        :param axis: Axis to maximize over.
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def minimum(self, x, y, out):
        """
        min(x, y)

        :param x:
        :param y:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def multiply(self, x, y, out):
        """
        x*y

        :param x:
        :param y:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def negative(self, x, out):
        """
        -x

        :param x:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def not_equal(self, x, y, out):
        """
        x != y
        :param x:
        :param y:
        :param out: Boolean tensor.
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def onehot(self, idx, out):
        """

        :param idx: Index tensor
        :param out: 2-d tensor, axis 0 gets onehot expansion
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reciprocal(self, x, out):
        """
        1/x

        :param x:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sign(self, x, out):
        """
        signum(x)

        :param x:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sin(self, x, out):
        """
        sine(x)

        :param x:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sqrt(self, x, out):
        """
        sqrt(x)

        :param x:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def square(self, x, out):
        """
        x^2

        :param x:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def subtract(self, x, y, out):
        """
        x - y

        :param x:
        :param y:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sum(self, x, axis, out):
        """
        sum of x over axis

        :param x:
        :param axis:
        :param out:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def tanh(self, x, out):
        """
        tanh(x)

        :param x:
        :param out:
        :return:
        """
        raise NotImplementedError()

    # @abc.abstractmethod
    def allreduce(self, x, out):
        """
        MPI allreduce

        :param x:
        :param out:
        :return:
        """
        raise NotImplementedError()

    # @abc.abstractmethod
    def conv2d(self, x, y, out):
        """
        2 dimensional convolution

        :param x:
        :param y:
        :param out:
        :return:
        """
        raise NotImplementedError()
