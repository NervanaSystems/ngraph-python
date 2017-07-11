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
import abc
from future.utils import with_metaclass
from collections import Iterable

from ngraph.op_graph.op_graph import SequentialOp, TensorValueOp, Op


class OpAccessor(with_metaclass(abc.ABCMeta, object)):
    """
    Provides access to some op properties when they may have been modified during passes.

    This is currently used so that the same pass can be used with op-graph and exec-graph if
    the pass uses the OpAccessor methods to access the components of the Op.
    """
    def __init__(self, **kwargs):
        self.replacement_list = []
        self.replacements = dict()

    @abc.abstractmethod
    def op_arg(self, op, n):
        """
        Returns the nth argument of an op-graph Op op as an op-graph Op.

        Overridden by the exec graph to reflect modifications made to the graph.

        Args:
            op: The op-graph op we want an args for.
            n: The arg number.

        Returns:
            The arg's op.

        """

    @abc.abstractmethod
    def op_args(self, op):
        """
        Returns all the arguments of an op-graph Op.

        Overridden by the exec graph to reflect modification made to the graph.

        Args:
            op: An op-graph Op.

        Returns:
            The args for op.

        """

    @abc.abstractmethod
    def get_device_op(self, op):
        """
        Helper function that traverses through any reshape ops or value ops
        to return the tensor op.

        Overridden by the exec graph to reflect modification made to the graph.

        Args:
            op: An op-graph Op.

        Returns:
            The op providing actual storage for op's value.

        """

    @abc.abstractmethod
    def run_pass(self, process_op, **kwargs):
        """
        Runs a pass to completion, calling process_op on each relevant op.

        """

    def begin_batch(self):
        """
        Called before beginning processing on a pass.
        """
        self.replacement_list = []

    def replace_op(self, op, replacement):
        """
        Queue op-graph Op op to be replaced by replacement at the end of the batch.

        Args:
            op: The op-graph Op being replaced.
            replacement: The replacement op-graph Op fro old_op.

        """
        self.replacement_list.append((op, replacement))

    @abc.abstractmethod
    def perform_replace_op(self, op, replacement):
        """
        Actually perform the op replacement

        Args:
            op: An Op to be replaced.
            replacement: An Op to replace op with.

        """

    def end_batch(self):
        """
        Called after a pass has been processed.

        Returns:
            True if the graph was changed.
        """
        for op, replacement in self.replacement_list:
            self.perform_replace_op(op, replacement)
            self.replacements[op] = replacement
        return len(self.replacement_list) > 0

    def get_replacement(self, op):
        return self.replacements.get(op, None)


class OpGraphOpAccessor(OpAccessor):
    """
    Provides access to some op properties when they may have been modified during passes.
    """
    def op_arg(self, op, n):
        """
        Returns the nth argument of an op-graph Op op as an op-graph Op.

        Overridden by the exec graph to reflect modifications made to the graph.

        Args:
            op: The op-graph op we want an args for.
            n: The arg number.

        Returns:
            The arg's op.

        """
        return self.op_args(op)[n]

    def op_args(self, op):
        """
        Returns all the arguments of an op-graph Op.

        Overridden by the exec graph to reflect modification made to the graph.

        Args:
            op: An op-graph Op.

        Returns:
            The args for op.

        """
        return op.args

    def get_device_op(self, op):
        """
        Helper function that traverses through any reshape ops or value ops
        to return the tensor op.

        Overridden by the exec graph to reflect modification made to the graph.

        Args:
            op: An op-graph Op.

        Returns:
            The op providing actual storage for op's value.

        """
        while isinstance(op, SequentialOp):
            op = op.value_tensor

        if op.is_device_op:
            return op

        if isinstance(op, TensorValueOp):
            return op.tensor

        for arg in op.args:
            dev_op = self.get_device_op(arg)
            if dev_op:
                return dev_op

        return None

    def run_pass(self, process_op, ops, **kwargs):
        assert isinstance(ops, Iterable), "Ops passed into do_pass must be an iterable"
        has_work = True
        while has_work:
            self.begin_batch()

            # pass through the ops in an execution order collecting things to do
            ops = Op.ordered_ops(op.forwarded for op in ops)
            for op in ops:
                op.update_forwards()
                process_op(op)

            has_work = self.end_batch()
            ops = list(op.forwarded for op in ops)

    def perform_replace_op(self, op, replacement):
        op.forwarded.replace_self(replacement.forwarded)


op_graph_op_accessor = OpGraphOpAccessor()


class DelegateOpAccessor(OpAccessor):
    """
    Delegates access to Op properties to op_accessor, which defaults to the op-graph accessor.
    """
    def __init__(self, op_accessor=op_graph_op_accessor):
        self.op_accessor = op_accessor

    def begin_pass(self, op_accessor=None, **kwargs):
        """
        Called before do_pass to peform pass initialization.

        Args:
            op_accessor: An OpAccessor for delegation.
            **kwargs:
        """
        if op_accessor is not None:
            self.op_accessor = op_accessor

    def end_pass(self, **kwargs):
        """
        Called after do_pass to perform any cleanup.

        Args:
            **kwargs:
        """
        pass

    def op_arg(self, op, n):
        """
        Returns the nth argument of an op-graph Op op as an op-graph Op.

        Overridden by the exec graph to reflect modifications made to the graph.

        Args:
            op: The op-graph op we want an args for.
            n: The arg number.

        Returns:
            The arg's op.

        """
        return self.op_accessor.op_arg(op, n)

    def op_args(self, op):
        """
        Returns all the arguments of an op-graph Op.

        Overridden by the exec graph to reflect modification made to the graph.

        Args:
            op: An op-graph Op.

        Returns:
            The args for op.

        """
        return self.op_accessor.op_args(op)

    def get_device_op(self, op):
        """
        Helper function that traverses through any reshape ops or value ops
        to return the tensor op.

        Overridden by the exec graph to reflect modification made to the graph.

        Args:
            op: An op-graph Op.

        Returns:
            The op providing actual storage for op's value.

        """
        return self.op_accessor.get_device_op(op)

    def run_pass(self, process_op, **kwargs):
        self.op_accessor.run_pass(process_op, **kwargs)

    def begin_batch(self):
        self.op_accessor.begin_batch()

    def replace_op(self, op, replacement):
        self.op_accessor.replace_op(op, replacement)

    def perform_replace_op(self, op, replacement):
        self.op_accessor.perform_replace_op(op, replacement)

    def end_batch(self):
        return self.op_accessor.end_batch()

    def get_replacement(self, op):
        return self.op_accessor.get_replacement(op)


class OpDelegate(with_metaclass(abc.ABCMeta, object)):
    def op_arg(self, op, n):
        """
        Returns the nth argument of an op-graph Op op as an op-graph Op.

        Overridden by the exec graph to reflect modifications made to the graph.

        Args:
            op: The op-graph op we want an args for.
            n: The arg number.

        Returns:
            The arg's op.

        """
        return self.op_args(op)[n]

    def op_args(self, op):
        """
        Returns all the arguments of an op-graph Op.

        Overridden by the exec graph to reflect modification made to the graph.

        Args:
            op: An op-graph Op.

        Returns:
            The args for op.

        """
        return op.args

    def get_device_op(self, op):
        """
        Helper function that traverses through any reshape ops or value ops
        to return the tensor op.

        Overridden by the exec graph to reflect modification made to the graph.

        Args:
            op: An op-graph Op.

        Returns:
            The op providing actual storage for op's value.

        """
        while isinstance(op, SequentialOp):
            op = op.value_tensor

        if op.is_device_op:
            return op

        if isinstance(op, TensorValueOp):
            return op.tensor

        for arg in op.args:
            dev_op = self.get_device_op(arg)
            if dev_op:
                return dev_op

        return None
