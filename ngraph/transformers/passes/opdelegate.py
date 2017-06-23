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
from ngraph.op_graph.op_graph import SequentialOp, TensorValueOp, Op


class OpAccessor(with_metaclass(abc.ABCMeta, object)):
    """
    Provides access to some op properties when they may have been modified during passes.

    This is currently used so that the same pass can be used with op-graph and exec-graph if
    the pass uses the OpAccessor methods to access the components of the Op.
    """
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
    def run_pass(self, process_op):
        """
        Runs a pass to completion, calling process_op on each relevant op.

        """

    @abc.abstractmethod
    def begin_batch(self):
        """
        Called before beginning processing on a pass.
        """

    @abc.abstractmethod
    def replace_op(self, op, replacement):
        """
        Queue op-graph Op op to be replaced by replacement at the end of the batch.

        Args:
            op: The op-graph Op being replaced.
            replacement: The replacement op-graph Op fro old_op.

        """

    @abc.abstractmethod
    def end_batch(self):
        """
        Called after a pass has been processed.

        Returns:
            True if the graph was changed.
        """


class OpGraphOpAccessor(OpAccessor):

    def __init__(self, **kwargs):
        super(OpGraphOpAccessor, self).__init__(**kwargs)
        self.replacement_list = []

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

    def run_pass(self, process_op, min_ops):
        has_work = True
        while has_work:
            self.begin_batch()

            # pass through the ops in an execution order collecting things to do
            ops = Op.ordered_ops(op.forwarded for op in min_ops)
            for op in ops:
                op.update_forwards()
                process_op(op)

            has_work = self.end_batch()
            min_ops = list(op.forwarded for op in min_ops)

    def begin_batch(self):
        self.replacement_list = []

    def replace_op(self, op, replacement):
        self.replacement_list.append((op, replacement))

    def end_batch(self):
        for old, rep in self.replacement_list:
            old.forwarded.replace_self(rep.forwarded)
        return len(self.replacement_list) > 0


op_graph_op_accessor = OpGraphOpAccessor()


class DelegateOpAccessor(OpAccessor):
    """
    Delegates access to Op properties to op_accessor, which defaults to the op-graph accessor.
    """
    def __init__(self, op_accessor=op_graph_op_accessor):
        self.op_accessor = op_accessor

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

    def end_batch(self):
        return self.op_accessor.end_batch()


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
