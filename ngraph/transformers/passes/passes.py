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
from ngraph.op_graph.op_graph import BroadcastOp, broadcast, DotOp, ReductionOp, make_axes, \
    axes_with_order, flatten_at, Transpose, unflatten, ReorderAxes, ContiguousOp, \
    OneHotTwoDimOp, BinaryElementWiseAxesOp, AssignOp, DotOneDimensional, DotTwoDimensional, \
    DotTwoByOne, ExpOp, LogOp, NegativeOp, OneHotOp, AssignOneDOp, ReshapeOp, flatten, constant, \
    Multiply, Add, Divide, Op, Sum, Prod, UnaryElementwiseAxesOp, \
    negative, cast_axes, power, DerivOp, ComputationOp

from ngraph.util.generics import generic_method
from ngraph.util.ordered import OrderedSet


class GraphPass(with_metaclass(abc.ABCMeta, object)):
    def __init__(self):
        super(GraphPass, self).__init__()

    @abc.abstractmethod
    def do_pass(self, ops):
        pass


class PeepholeGraphPass(GraphPass):
    def __init__(self):
        super(PeepholeGraphPass, self).__init__()

    def find_initializers(self, ops):
        did_work = False
        ops = Op.ordered_ops(ops)
        while True:
            new_inits = OrderedSet()
            for op in ops:
                if op.initializers:
                    new_inits.update(op.initializers)
            for init in new_inits:
                init = init.forwarded
                if init not in self.inits:
                    self.inits = OrderedSet(Op.ordered_ops(new_inits + self.inits))
                    ops = self.inits
                    did_work = True
                    break
            else:
                return did_work

    def do_pass(self, ops, inits):
        self.inits = inits
        assert isinstance(ops, Iterable), "Ops passed into do_pass must be an iterable"
        has_work = True
        while has_work or self.find_initializers(ops):
            self.replacement_list = []
            ops = OrderedSet(op.forwarded for op in self.inits + ops)
            for op in Op.ordered_ops(ops):
                op.update_forwards()
                self.visit(op)
            for old, rep in self.replacement_list:
                old.forwarded.replace_self(rep.forwarded)
            has_work = len(self.replacement_list) > 0
        return ops, self.inits

    def replace_op(self, op, replacement):
        """
        TODO.

        Arguments:
          op: TODO
          replacement: TODO

        Returns:
          TODO
        """
        self.replacement_list.append((op, replacement))


class RequiredTensorShaping(PeepholeGraphPass):
    """TODO."""
    @generic_method(dispatch_base_type=Op)
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO
        """
        pass

    @visit.on_type(ReductionOp)
    def visit(self, op):
        if op.must_reduce:
            self.replace_op(op, op.reduce_to_twod())

    @visit.on_type(DotOp)
    def visit(self, op):
        x, y = op.args
        x_reduction_axes = op.x_reduction_axes
        y_reduction_axes = op.y_reduction_axes
        out_axes = op.axes
        if len(x_reduction_axes) == 0:
            d = make_axis(1)
            x_reduction_axes = make_axes((d,))
            y_reduction_axes = x_reduction_axes
            x = broadcast(x, x.axes + x_reduction_axes)
            y = broadcast(y, y_reduction_axes + y.axes)

        if x.is_scalar:
            temp = x
            x = y
            y = temp
        if y.is_scalar:
            if x.is_scalar:
                out = x.scalar_op * y.scalar_op
                if len(x_reduction_axes) > 0:
                    out = out * x_reduction_axes.size
                out = broadcast(out, op.axes)
            else:
                out = Sum(x, x_reduction_axes) * y.scalar_op
            out = broadcast(out, op.axes)
        else:
            x_rem_axes = x.axes - x_reduction_axes
            x = axes_with_order(x, x_rem_axes + x_reduction_axes)

            y_rem_axes = y.axes - y_reduction_axes
            y = axes_with_order(y, y_reduction_axes + y_rem_axes)

            x = flatten_at(x, len(x.axes) - len(x_reduction_axes))
            y = flatten_at(y, len(y_reduction_axes))

            if len(out_axes) == 0:
                out = DotOneDimensional(x, y, axes=())
            elif len(x.axes) == 1:
                y = Transpose(y)
                out = DotTwoByOne(y, x, axes=y.axes[0])
            elif len(y.axes) == 1:
                out = DotTwoByOne(x, y, axes=x.axes[0])
            else:
                out = DotTwoDimensional(x, y,
                                        axes=([op.x_out_axes.flatten(True),
                                               op.y_out_axes.flatten(True)]))

            out = unflatten(out)
            out = ReorderAxes(out, out_axes)

        self.replace_op(op, out)

    @visit.on_type(DotOneDimensional)
    def visit(self, op):
        pass

    @visit.on_type(DotTwoDimensional)
    def visit(self, op):
        pass

    @visit.on_type(DotTwoByOne)
    def visit(self, op):
        pass

    @visit.on_type(Sum)
    def visit(self, op):
        x = op.args[0]
        if x.is_scalar:
            # Sum of a scalar is just the scalar times the axes size rebroadcast
            val = broadcast(cast_axes(x, ()) * op.reduction_axes.size, op.axes)
            self.replace_op(op, val)
            return
        # call-next-method
        if op.must_reduce:
            self.replace_op(op, op.reduce_to_twod())

    @visit.on_type(Prod)
    def visit(self, op):
        x = op.args[0]
        if x.is_scalar:
            # Prod of a scalar is just the scalar raised to the power of the
            # axes size rebroadcast
            val = broadcast(power(cast_axes(x, ()), op.reduction_axes.size), op.axes)
            self.replace_op(op, val)
            return
        # call-next-method
        if op.must_reduce:
            self.replace_op(op, op.reduce_to_twod())

    @visit.on_type(OneHotOp)
    def visit(self, op):
        self.replace_op(op, op.as_two_dim())

    @visit.on_type(OneHotTwoDimOp)
    def visit(self, op):
        pass

    @visit.on_type(UnaryElementwiseAxesOp)
    def visit(self, op):
        self.replace_op(op, op.reduce_to_one_d())

    @visit.on_type(BinaryElementWiseAxesOp)
    def visit(self, op):
        self.replace_op(op, op.reduce_to_oned())

    @visit.on_type(ContiguousOp)
    def visit(self, op):
        if op.args[0].tensor_description().c_contiguous:
            self.replace_op(op, op.args[0])

    @visit.on_type(AssignOp)
    def visit(self, op):
        tensor, val = op.args
        assert not isinstance(tensor, ReshapeOp)
        tensor, val = flatten(tensor), flatten(val)
        self.replace_op(op, AssignOneDOp(tensor, val, force=op.force))

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


class DerivPass(PeepholeGraphPass):
    """
    The pass that computes derivatives, i.e. expanding DerivOp to actual
    derivatives.
    """

    @staticmethod
    def _deriv(dependent, independent, error=constant(1)):
        """
        Computes the operation for [dDependent/dIndependent](error=1).
        The derivative is a multi-linear function.
        Args:
            dependent (TensorOp): Dependent op.
            independent(TensorOp): Independent op.
            error (TensorOp, optional): The tensor holding the error where the
                derivative will be computed at. Must have the same axes as dependent.
        Returns:
            TensorOp: Derivative applied to error. Has axes of independent.
        """
        if not error.axes.has_same_axes(dependent.axes):
            raise ValueError(
                "Dependent and error must have the same set of axes")

        adjoints = dependent.forwarded.adjoints(error)

        if independent.forwarded not in adjoints:
            return constant(0, independent.axes)

        adjoint = adjoints[independent.forwarded]
        return broadcast(adjoint.forwarded, axes=independent.axes)

    @generic_method()
    def visit(self, op):
        pass

    @visit.on_type(DerivOp)
    def visit(self, op):
        # redundant names to keep consistent for now
        deriv = DerivPass._deriv(op.dependent, op.independent, op.error)
        self.replace_op(op, deriv)


class CompUserDepsPass(PeepholeGraphPass):
    """
    Pass that converts ComputationOp's user_deps. Currently CompUserDepsPass
    is required since passes are not able to add ops that require user_deps,
    which may need to be refactored.

    TODO: This is a temporary fix until user_deps gets cleaned up.
    """

    @generic_method()
    def visit(self, op):
        pass

    @visit.on_type(ComputationOp)
    def visit(self, op):
        op.require_user_deps(list(map(lambda x: x.forwarded, op.other_deps)))


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
