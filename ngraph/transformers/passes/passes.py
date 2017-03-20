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

from ngraph.op_graph.axes import make_axis
from ngraph.op_graph.op_graph import BroadcastOp, broadcast, DotOp, make_axes, \
    axes_with_order, flatten_at, Transpose, unflatten, ReorderAxes, \
    ContiguousOp, DotLowDimension, \
    ExpOp, LogOp, NegativeOp, constant, \
    Multiply, Add, Divide, Op, Sum, Prod, negative, power

from ngraph.util.generics import generic_method


class GraphPass(with_metaclass(abc.ABCMeta, object)):

    @abc.abstractmethod
    def do_pass(self, ops, transformer):
        pass


class GraphBuildingPass(GraphPass):
    """
    Base class for passes that build new graph, primarily derivatives
    and other macro-like things.

    """
    def do_pass(self, min_ops, transformer):
        """
        Visit the ops until nothing changes.

        Args:
            min_ops: The set of ops that must be computed.
            transformer: An InitGraph object.

        """
        assert isinstance(min_ops, Iterable), "Ops passed into do_pass must be an iterable"
        has_work = True
        while True:
            if not has_work:
                return

            self.replacement_list = []

            # pass through the ops in an execution order collecting things to do
            ops = Op.ordered_ops(op.forwarded for op in min_ops)
            for op in ops:
                op.update_forwards()
                self.visit(op)

            # Perform the gathered replacements
            for old, rep in self.replacement_list:
                old.forwarded.replace_self(rep.forwarded)
            has_work = len(self.replacement_list) > 0
            min_ops = list(_.forwarded for _ in min_ops)

    def replace_op(self, op, replacement):
        """
        Replace op with replacement.

        Args:
            op: op to be replaced.
            replacement: new op.

        """
        self.replacement_list.append((op, replacement))


class PeepholeGraphPass(GraphBuildingPass):
    """
    Base class for passes that do not add to the graph.
    """
    def do_pass(self, min_ops, transformer):
        """
        Visit the ops until nothing changes.

        Args:
            min_ops: The set of ops that must be computed.
            transformer: An InitGraph object.

        """
        assert isinstance(min_ops, Iterable), "Ops passed into do_pass must be an iterable"
        has_work = True
        while True:
            if not has_work:
                return

            # pass through the ops in an execution order collecting things to do
            self.replacement_list = []
            ops = Op.ordered_ops(op.forwarded for op in min_ops)
            for op in ops:
                op.update_forwards()
                self.visit(op)
            for old, rep in self.replacement_list:
                old.forwarded.replace_self(rep.forwarded)
            has_work = len(self.replacement_list) > 0
            min_ops = list(_.forwarded for _ in min_ops)


class RequiredTensorShaping(PeepholeGraphPass):
    """
    Tensor shaping pass common to gpu and cpu.
    Currently used in DotOp.
    """

    @generic_method(dispatch_base_type=Op)
    def visit(self, op):
        pass

    @visit.on_type(DotOp)
    def visit(self, op):
        x, y = op.args
        reduction_axes = op.reduction_axes
        out_axes = op.axes
        if len(reduction_axes) == 0:
            # TODO: this is a weird case, should we really support it?
            d = make_axis(1)
            reduction_axes = make_axes((d,))
            x = broadcast(x, x.axes + reduction_axes)
            y = broadcast(y, reduction_axes + y.axes)

        if x.is_scalar:
            x, y = y, x

        if y.is_scalar:
            if x.is_scalar:
                out = x.scalar_op * y.scalar_op
                if len(reduction_axes) > 0:
                    out = out * reduction_axes.size
                out = broadcast(out, op.axes)
            else:
                out = Sum(x, reduction_axes) * y.scalar_op
            out = broadcast(out, op.axes)
        else:
            # move reduction_axes to end
            x = axes_with_order(x, (x.axes - reduction_axes) + reduction_axes)
            # move reduction axes to front
            y = axes_with_order(y, reduction_axes + (y.axes - reduction_axes))

            # flatten non-reduction axes together and reduction axes together
            x = flatten_at(x, len(x.axes) - len(reduction_axes))
            # flatten non-reduction axes together and reduction axes together
            y = flatten_at(y, len(reduction_axes))

            if len(out_axes) == 0:
                out = DotLowDimension(x, y, axes=())
            elif len(x.axes) == 1:
                y = Transpose(y)
                out = DotLowDimension(y, x, axes=y.axes[0])
            elif len(y.axes) == 1:
                out = DotLowDimension(x, y, axes=x.axes[0])
            else:
                out = DotLowDimension(x, y, axes=([op.x_out_axes.flatten(True),
                                                   op.y_out_axes.flatten(True)]))

            out = unflatten(out)
            out = ReorderAxes(out, out_axes)

        self.replace_op(op, out)


class CPUTensorShaping(PeepholeGraphPass):
    """TODO."""
    @generic_method(dispatch_base_type=Op)
    def visit(self, op):
        pass

    @visit.on_type(ContiguousOp)
    def visit(self, op):
        if op.args[0].tensor_description().c_contiguous:
            self.replace_op(op, op.args[0])

    @visit.on_type(ReorderAxes)
    def visit(self, op):
        x = op.args[0]
        if op.axes == x.axes:
            self.replace_op(op, x)

    @visit.on_type(BroadcastOp)
    def visit(self, op):
        x = op.args[0]
        if op.axes == x.axes:
            self.replace_op(op, x)


class SimplePrune(PeepholeGraphPass):
    """TODO."""
    @generic_method()
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO
        """
        pass

    @visit.on_type(NegativeOp)
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
        x, = op.args
        if x.is_scalar and x.is_constant:
            self.replace_op(op, constant(-x.const))

    @visit.on_type(Multiply)
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
        x, y = op.args
        rep = None
        if x.is_scalar and x.is_constant:
            if x.const == 0:
                rep = x
            elif x.const == 1:
                rep = y
            elif x.const == -1:
                rep = negative(y)
        elif y.is_scalar and y.is_constant:
            if y.const == 0:
                rep = y
            elif y.const == 1:
                rep = x
            elif y.const == -1:
                rep = negative(x)
        if rep is not None:
            self.replace_op(op, rep)

    @visit.on_type(Add)
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
        x, y = op.args
        rep = None
        if x.is_scalar and x.is_constant:
            if x.const == 0:
                rep = y
        elif y.is_scalar and y.is_constant:
            if y.const == 0:
                rep = x
        if rep is not None:
            self.replace_op(op, rep)

    @visit.on_type(Sum)
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
        x, = op.args
        if x.is_scalar and x.is_constant:
            val = x.const * op.reduction_axes.size
            self.replace_op(op, constant(val))

    @visit.on_type(Prod)
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
        x, = op.args
        if x.is_scalar and x.is_constant:
            val = power(x.const, op.reduction_axes.size)
            self.replace_op(op, constant(val))

    @visit.on_type(LogOp)
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
        x, = op.args
        if isinstance(x, Divide):
            num, denom = x.args
            if isinstance(num, ExpOp):
                exp_x, = num.args
                self.replace_op(op, exp_x - type(op)(denom))
        elif isinstance(x, ExpOp):
            exp_x, = x.args
            self.replace_op(op, exp_x)
