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
import abc

from future.utils import with_metaclass

from ngraph.transformers.passes.passes import PeepholeGraphPass, GraphPass
from ngraph.util.generics import generic_method
from ngraph.op_graph.op_graph import Op, ContiguousOp


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

    @abc.abstractmethod
    def __str__(self):
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
        If no layout transform is needed, this should return 0. Otherwise it returns a cost
        value for the layout transform required.
        """
        pass

    @abc.abstractmethod
    def get_layout_transform(self, arg_layout, op_layout, arg):
        """
        If no layout transform is needed, this should return None. Otherwise it returns an op
        which can replace the arg in the graph to satisfy the layout constraint. An example
        would be a device specific dimshuffle of the arg.
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


class PruneContiguousPass(PeepholeGraphPass):
    """
    Temporary pass to remove contiguous ops from the graph before doing layout.
    TODO: stop inserting contiguous ops? Need to handle other transformer reqs though
    """
    @generic_method(dispatch_base_type=Op)
    def visit(self, op):
        pass

    @visit.on_type(ContiguousOp)
    def visit(self, op):
        self.replace_op(op, op.args[0])


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
        self.users = dict()

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
            self.unary_constraints[op] = self.transformer.get_layout_cost_function(op)

            # Find all args that are device ops and generate binary constraints
            # Binary constraints map each op to a list of tuples storing (argument, constraint)
            self.binary_constraints[op] = []
            for arg in op.args:
                arg_op = self.get_device_op(arg)
                if arg_op:
                    self.binary_constraints[op].append(
                        (arg_op, self.transformer.get_layout_change_cost_function(op, arg)))

                    # Add this op to the arg's users to construct digraph
                    if arg_op not in self.users:
                        self.users[arg_op] = [op]
                    else:
                        self.users[arg_op].append(op)


class AssignLayouts(GraphPass):
    """
    Computes an upper bound for layout cost by using default layouts for every op, then
    attempts to minimize the WCSP until termination criteria are met.
    """
    def __init__(self, domain_pass, constraint_pass):
        self.domain_pass = domain_pass
        self.constraint_pass = constraint_pass

        self.domains = None
        self.unary_constraints = None
        self.binary_constraints = None
        self.users = None

    def compute_default_cost(self):
        cost = 0.0
        assignment = dict()

        for op in self.domains:
            # Default layout is the first in the domain
            op_layout = self.domains[op][0]
            assignment[op] = op_layout
            cost += self.unary_constraints[op].get_cost(op_layout)

            # Get all argument default layouts and any transition costs
            for arg_op, constraint in self.binary_constraints[op]:
                arg_layout = self.domains[arg_op][0]
                cost += constraint.get_cost(arg_layout, op_layout)

        return (assignment, cost)

    def compute_op_assignment_cost(self, op, layout, assignment):
        cost = self.unary_constraints[op].get_cost(layout)

        # Compute costs for constraints to each of this op's arguments
        for arg_op, constraint in self.binary_constraints[op]:
            if arg_op in assignment and assignment[arg_op]:
                cost = cost + constraint.get_cost(layout, assignment[arg_op])

        # Compute costs for any ops which use this op as an argument
        if op in self.users:
            for user in self.users[op]:
                if user in assignment and assignment[user]:
                    # Find constraint matching this pair (user, op)
                    for arg_op, constraint in self.binary_constraints[user]:
                        if arg_op is op:
                            cost = cost + constraint.get_cost(assignment[user], layout)
                            break

        return cost

    def branch_and_bound(self, cur_assignment, unassigned, cost, min_assignment, upper_bound):
        """
        Uses depth first branch and bound to find the minimum cost layout assignment.
        This is not useful in most cases, because it is O(d^n) where d is domain size (layouts
        per op) and n is number of nodes. Even for a small model like MNIST MLP it takes hours.
        """
        # If all ops are assigned, this is the new minimum cost assignment
        if not unassigned:
            return (cur_assignment.copy(), cost)

        # Choose an op to assign and remove from unassigned set
        new_unassigned = set(unassigned)
        op = new_unassigned.pop()

        # Iterate over layout domain of the op
        for layout in self.domains[op]:
            assignment_cost = self.compute_op_assignment_cost(op, layout, cur_assignment)

            # Prune sub-tree if cost is more than bound
            if (cost + assignment_cost) < upper_bound:
                # Continue searching with this assignment
                cur_assignment[op] = layout
                min_assignment, upper_bound = self.branch_and_bound(cur_assignment,
                                                                    new_unassigned,
                                                                    cost + assignment_cost,
                                                                    min_assignment,
                                                                    upper_bound)

        # Remove assignment for this op to step up in search tree
        cur_assignment[op] = None

        return (min_assignment, upper_bound)

    def minimize_cost(self, min_assignment, upper_bound):
        """
        Run depth first branch and bound search for minimum cost layout assignment,
        initially bounded by default layout cost
        """
        cur_assignment = dict()
        unassigned = set(self.domains.keys())

        return self.branch_and_bound(cur_assignment, unassigned, 0, min_assignment, upper_bound)

    def do_pass(self, ops, transformer):
        # Initialize data needed for layout optimization
        self.domains = self.domain_pass.domains
        self.unary_constraints = self.constraint_pass.unary_constraints
        self.binary_constraints = self.constraint_pass.binary_constraints
        self.users = self.constraint_pass.users

        # Use default layouts to compute upper bound for cost
        # TODO: implement heuristic optimizer(s)
        self.min_assignment, upper_bound = self.compute_default_cost()

        # Assign layouts to each tensor
        for op in self.min_assignment:
            self.min_assignment[op].set_shape_strides()
            op.metadata["layout"] = self.min_assignment[op]


class AddLayoutConversions(PeepholeGraphPass):
    """
    Inserts layout conversions into the graph as needed based on assigned layouts
    """
    def __init__(self, assign_pass):
        self.assign_pass = assign_pass
        self.binary_constraints = None

    def do_pass(self, ops, transformer):
        self.binary_constraints = self.assign_pass.binary_constraints
        super(AddLayoutConversions, self).do_pass(ops, transformer)

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
        if "layout" in op.metadata:
            new_args = []
            for arg in op.args:
                b_constraint = None
                dev_op = self.get_device_op(arg)
                orig_arg_op = None
                if dev_op is None:
                    new_args.append(arg)
                    continue

                # Find matching constraint
                for arg_op, constraint in self.binary_constraints[op]:
                    if arg_op.forwarded is dev_op:
                        b_constraint = constraint
                        orig_arg_op = arg_op
                        break

                # Get layout conversion ops for this arg
                if b_constraint is not None:
                    new_args.append(b_constraint.get_layout_transform(orig_arg_op.metadata["layout"],
                                                                      op.metadata["layout"],
                                                                      arg))
                else:
                    new_args.append(arg)

            # Replace op if any inputs need to be transformed
            if any(a is not b for a,b in zip(new_args, list(op.args))):
                op_type = type(op)
                new_op = op_type(*new_args)

                self.replace_op(op, new_op)
                self.binary_constraints[new_op] = self.binary_constraints[op]
