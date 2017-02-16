# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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

from ngraph.transformers.passes.passes import PeepholeGraphPass, GraphPass
from ngraph.util.generics import generic_method
from ngraph.op_graph.op_graph import Op


class LayoutAssignment(with_metaclass(abc.ABCMeta, object)):
    """
    Base class for device specific layout. Defines how a tensor with an arbitrary
    number of axes is stored in device memory. Individual transformers must sub-class
    this with their own layout specifications.

    This corresponds to an assignment value in a weighted constraint satisfaction problem (WCSP).
    A collection of these makes up the domain for a variable.
    """
    def __init__(self):
        pass


class BinaryLayoutConstraint(with_metaclass(abc.ABCMeta, object)):
    """
    Base class for device specific binary layout constraint. Each device may impose constraints
    between the output layout of an argument and the layout of the op it feeds. This may vary
    based on the op and the available device implementations of that op.

    This corresponds to a binary soft weighted constraint in a WCSP.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_cost(self, arg_layout, op_layout):
        """
        If the constraint is fully satisfied, this should return 0. Otherwise this will
        return the cost of violating the constraint given two values.
        """
        pass


class UnaryLayoutConstraint(with_metaclass(abc.ABCMeta, object)):
    """
    Base class for device specific unary layout constraint. This kind of constraint can be
    used to impose a cost function on an individual op's layout.

    This corresponds to a unary soft weighted constraint in a WCSP.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_cost(self, op_layout):
        """
        Returns a cost for using this layout for the given op.
        """
        pass


class GenerateLayoutDomains(PeepholeGraphPass):
    """
    This pass generates domains (possible layouts) for each variable (op) in the graph
    """
    def __init__(self, transformer):
        self.transformer = transformer
        self.domains = dict()

    @generic_method(dispatch_base_type=Op)
    def visit(self, op):
        if op.is_device_op:
            self.domains[op] = self.transformer.get_layouts(op)


class GenerateLayoutConstraints(PeepholeGraphPass):
    """
    This pass generates unary and binary constraints for each op. Binary constraints are
    generated for (op, arg) pairs when visiting the op.
    """
    def __init__(self, transformer):
        self.transformer = transformer
        self.unary_constraints = dict()
        self.binary_constraints = dict()

    def get_device_op(self, op):
        if op.is_device_op:
            return op

        for arg in op.args:
            dev_op = self.get_device_op(arg)
            if dev_op:
                return dev_op

        return None

    @generic_method(dispatch_base_type=Op)
    def visit(self, op):
        if op.is_device_op:
            # Generate unary constraint by getting the cost function for this op
            self.unary_constraints[op] = transformer.get_layout_cost_function(op)

            # Find all args that are device ops and generate binary constraints
            for arg in op.args:
                arg_op = self.get_device_op(arg)
                if arg_op:
                    self.binary_constraints[{op, arg_op}] = transformer.get_layout_change_cost_function(op, arg)


class AssignLayouts(GraphPass):
    """TODO."""
    def do_pass(self, ops, transformer):
        # Use default layouts to compute upper bound for cost

        # Run branch and bound algorithm to look for better layout assignments
