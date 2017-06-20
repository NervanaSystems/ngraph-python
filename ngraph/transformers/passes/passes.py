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
import itertools

from future.utils import with_metaclass
from collections import Iterable

from ngraph.op_graph.axes import make_axis
from ngraph.op_graph.op_graph import BroadcastOp, broadcast, DotOp, make_axes, \
    axes_with_order, flatten_at, Transpose, unflatten, ReorderAxes, \
    ContiguousOp, DotLowDimension, \
    ExpOp, LogOp, NegativeOp, constant, \
    Multiply, Add, Divide, Op, Sum, Prod, negative, power, \
    PatternLabelOp, PatternSkipOp, DelegateOpAccessor

from ngraph.util.generics import generic_method


class GraphPass(with_metaclass(abc.ABCMeta, DelegateOpAccessor)):
    @abc.abstractmethod
    def do_pass(self, min_ops, transformer):
        pass


class ProcessOpGraphPass(GraphPass):
    def do_pass(self, min_ops, transformer):
        assert isinstance(min_ops, Iterable), "Ops passed into do_pass must be an iterable"
        self.run_pass(self.process_op, min_ops=min_ops)

    @abc.abstractmethod
    def process_op(self, op):
        """
        Called from run_pass to perform processing on an op-graph Op op.

        Args:
            op: The op-graph Op to be

        Returns:

        """


class GraphBuildingPass(ProcessOpGraphPass):

    def process_op(self, op):
        self.visit(op, *op.args)


# How to use the graph rewrite pass
#
# Graph rewrite pass essentially allows pass users to rewrite parts of the
# input graph in any way they want. Fusion is one example of graph rewrite that
# fuses multiple ops together. At a high-level users of the pass need to
# specify 2 things: 1) which ops to fuse, and 2) how to create new op(s) from
# the existing ops.
#
# In order to specify which ops to rewrite, users specify a 'pattern'. A
# pattern, in very basic terms, is a sub-graph that is to be rewritten. These
# patterns are constructed using the same ngraph APIs used in constructing
# actual graphs.  E.g., to construct Add op, use ng.Add(). Since the new op is
# most likely going to be constructed using some of the rewritten ops, the
# graph rewrite pass allows users to refer to the parts of the sub-graph by
# providing 'label' op (This is an op called PatternLabelOp in op_graph.) Label
# op follows the notion of binding variables to terms from functional
# languages.
#
# To give an example of a label usage, assume that we want to fuse addition of
# Conv2D and Bias into a single Conv2DWithBias op. Conv2DWithBias op will use
# same inputs as Conv2D op and Bias op, so we need to refer to the inputs of
# Conv2D op and Bias op to construct the fused op. In order to refer to the
# inputs, we would use labels. Essentially, the rule for fusion would
# conceptually look like: Add(Conv2D(X, Y), Bias(Z)) = Conv2DWithBias(X, Y, Z).
# Here X, Y, and Z are labels.
#
# A complication that often arises in ngraph's pattern matching is the presence
# of unanticipated ops such as BroadCast.  These ops are not expected by the
# user, but they are inserted in the graph to for certain operations, such as
# adjust axes as an example. To accommodate such ops, the graph rewrite pass
# provides PatternSkipOp, which is an op to specify to the pass that if such
# unanticipated op is found in the place of a PatternSkipOp, then skip the
# pattern match for it. The ops to be skipped are specified by the user by
# defining a predicate that holds true for ops to be skipped.
#
# Once the patterns are written, a user registers them with the graph rewrite
# pass using GraphRewritePass.register_pattern API. register_pattern API takes
# 2 inputs: a pattern that is to be searched in the graph, and the callback
# function to be invoked when the pattern matches. The callback function is the
# place where user would implement logic for rewriting sub-graphs. The callback
# function aptly gets label map and the matched op as inputs. The label map is
# simply a map of all the labels used in the pattern and the ops in the
# sub-graph that they map to. In the above example, if actual graph contained
# Add(Conv2D({1}, {2}), Bias({3})), then the label map would contain X={1},
# Y={2}, Z={3}. For examples of using API and label, refer to cpufusion.py that
# uses rewrite pass to implement fusion for forward and backward ReLU
# operation.
#
# Rewrite pass works as follows: 1) a user creates and registers all
# required<pattern, callback function> with the rewrite pass in the pass init
# routine, 2) when do_pass for the rewrite pass is invoked, it goes over the
# complete graph looking for matches for all the registered patterns, and for
# the matching patterns invokes their callback functions. This is a standard
# design borrowed from numerous places such as a specification of yacc rules
# (grammar rules).
#
# The reason for using this design is: since all patterns are known to the
# rewrite pass beforehand, it can scan whole graph only once, and match all the
# patterns.  An optimization that we have not yet implemented (we may not
# implement that for now since the number of patterns is too small) is
# constructing an FSA (automata) from all the patterns. Automata will ensure
# the fastest possible match for the patterns.  Even without automata, this
# design allows us to do such optimizations in the future.
#
#
# A word on ordering of the patterns: The order in which patterns are
# registered to the rewrite pass is the order in which the pass looks for their
# match. In other words, if the patterns match same sub-graph then the callback
# function of the one registered first would be invoked first. Also, it is not
# necessary that the callback functions must do the replacement -- it can
# simply be a debug pattern matching rule that prints the matching sub-graphs.
#


class GraphRewritePass(ProcessOpGraphPass):
    """
    A utility class for rewriting graph, including fusion

    """
    # Return values for pattern matching
    found = True
    not_found = False

    def __init__(self, **kwargs):
        super(GraphRewritePass, self).__init__(**kwargs)
        self.registered_patterns = []
        self.replacement_list = []

    def match_pattern_label_op(self, op, pattern, label_map):
        """
        Helper function used by match_pattern

        'pattern' is PatternLabelOp. Process 'op' accordingly and update
        label_map by binding the label if appropriate.

        Return:
          true, if op matches pattern; false otherwise
          label_map is updated with assignments in case of match

        """
        # If it is a label and the constraint for the label is satisfied,
        # then we store the mapping. But if the label is already bound to
        # some value, then we need to check that the value already assigned
        # to the label is same as what we are going to assign to it. This
        # way we are supporting equality constraints on the labels. A user
        # can say "X" in some subgraph must be same as "X" in other subgraph.
        if pattern.label in label_map:
            if label_map[pattern.label] == op and pattern.constraint_fn(op):
                return GraphRewritePass.found
            else:
                return GraphRewritePass.not_found

        elif pattern.constraint_fn(op):
            # Label is not already bound and constraint is satisfied, so bind
            # the label now.
            label_map[pattern.label] = op
            return GraphRewritePass.found

        else:
            return GraphRewritePass.not_found

    # Function to iterate over all args and match them
    def iterate_all_args(self, op_args, pattern_args, label_map):
        """
        Helper function to iterate over all args (children) of an op (parent)
        and check if args of op can be matched with args of pattern

        Return:
          true, if op matches pattern; false otherwise
          label_map is updated with assignments in case of match

        """
        for o_arg, p_arg in zip(op_args, pattern_args):
            # Depth-first graph traversal
            if self.match_pattern(o_arg,
                                  p_arg,
                                  label_map) == GraphRewritePass.not_found:
                return GraphRewritePass.not_found
        # If pattern for all args match, then we return true.
        return GraphRewritePass.found

    def match_op_args(self, op, pattern, label_map):
        """
        Helper function to match args (children) of an 'op' with args of 'pattern'

        This function considers commutavity of ops to match different possible
        orderings of args.

        Return:
          true, if op matches pattern; false otherwise
          label_map is updated with assignments in case of match

        """
        # For commutative op, check for all permutations of pattern args.
        if op.is_commutative:
            for pattern_args in itertools.permutations(pattern.args):
                # We need to make a copy of label_map because if a particular
                # permutation of arg ordering does not match, we need to throw
                # away all the updates made to the label_map for the failed ordering.
                new_label_map = dict(label_map)
                if self.iterate_all_args(self.op_args(op),
                                         pattern_args,
                                         new_label_map) == GraphRewritePass.found:
                    # If this ordering is successful, then we update label_map.
                    for k, v in new_label_map.items():
                        label_map[k] = v
                    return GraphRewritePass.found
            return GraphRewritePass.not_found
        else:
            # otherwise, just check for default ordering.
            # Since we only have 1 option for default ordering, we do not
            # need to make a copy of label_map.
            return self.iterate_all_args(self.op_args(op), pattern.args, label_map)

    def match_pattern(self, op, pattern, label_map):
        """
        Matches a pattern 'pattern' (sub-graph) in the sub-graph specified by 'op'
        Considers commutativity of ops which matching ops

        Args:
          op: sub-graph node where we should begin pattern match
          pattern: pattern specifying subgraph to be matched at op
          label_map: dictionary to be updated with sub-graph assignments
                     for labels in the pattern if match is found

        Returns"
          true, if op matches pattern; false otherwise
          label_map is updated with assignments in case of match

        """

        if isinstance(pattern, PatternLabelOp):
            return self.match_pattern_label_op(op, pattern, label_map)

        elif isinstance(pattern, PatternSkipOp) and not pattern.is_optional_op_fn(op):
            # If pattern contains SkipOp, but 'op' is not optional, then we
            # check if children of SkipOp satisfies 'op'. E.g., pattern contains
            # PatternSkipOp and BroadCast is optional, but 'op' is TensorValueOp.
            assert len(pattern.args) == 1, "PatternSkipOp only allows 1 input"
            return self.match_pattern(op, pattern.args[0], label_map)

        else:
            do_lengths_match = (len(self.op_args(op)) == len(pattern.args))
            is_same_type = (type(op) == type(pattern))
            is_skip_op = (isinstance(pattern, PatternSkipOp) and
                          pattern.is_optional_op_fn(op))

            do_explore_children = (is_same_type or is_skip_op) and do_lengths_match
            if do_explore_children:
                # We explore children of 'op' in 2 cases:
                #  1) if type of op is same as type of pattern (strict instance check)
                #  2) if type of pattern is SkipOp and 'op' is optional
                #     E.g., pattern contains PatternSkipOp and BroadCastOp is optional,
                #     and 'op' is BroadCastOp.
                return self.match_op_args(op, pattern, label_map)

            else:
                # If matching of ops failed, we return not_found.
                return GraphRewritePass.not_found

    @staticmethod
    def print_pattern(pattern, spaces=0):
        """
        Print pattern for debugging purpose

        """
        str = ""
        for t in range(0, spaces):
            str += ' '
        str += pattern.__str__()
        for arg in pattern.args:
            str += '\n'
            str += GraphRewritePass.print_pattern(arg, spaces + 1)
        return str

    def register_pattern(self, pattern, callback_fn):
        """
        Register a 'pattern' with callback function 'callback_fn'

        """
        self.registered_patterns.append((pattern, callback_fn))

    def do_pass(self, min_ops, transformer):
        """
        Visit the ops and do pattern matching and replacement until nothing changes.

        Args:
            min_ops: The set of ops to be checked for pattern match
            transformer: An InitGraph object.

        """
        assert isinstance(min_ops, Iterable), "Ops passed into do_pass must be an iterable"
        has_work = True
        while has_work:
            self.begin_batch()

            # pass through the ops in an execution order collecting things to do
            ops = Op.ordered_ops(op.forwarded for op in min_ops)
            for op in ops:
                op.update_forwards()
                self.process_op(op)

            # Perform the gathered replacements
            has_work = self.end_batch()
            min_ops = list(op.forwarded for op in min_ops)

    def process_op(self, op):
        # For performing pattern match, we have 2 options:
        # 1) for every graph node, check if any pattern match
        # 2) for every pattern, check if any graph node match
        # We choose 1st option. But we can also choose 2nd.
        # TODO(nhasabni) Two issues that complicate above choice is:
        #  1) A pattern may match multiple graph nodes
        #  2) Multiple patterns may match single graph node
        # These issues need to be discussed.

        # Iterate over all registered patterns and check for pattern match
        for pattern, callback_fn in self.registered_patterns:
            # list of (label_map, op) tuples that match pattern
            # Given pattern may match multiple times in the graph. For every
            # such match, we have label_map and the op that matches the
            # pattern. So we use list of tuples as return type.
            label_map_op_list = []
            label_map = dict()

            if self.match_pattern(op, pattern, label_map):
                label_map_op_list.append((label_map, op))
                callback_fn(op, label_map_op_list)


class PeepholeGraphPass(GraphBuildingPass):
    """
    Base class for passes that do not add to the graph.

    TODO: currently it has same exact implementation as GraphBuildingPass,
    consider removing it in future.
    """
    pass


class RequiredTensorShaping(PeepholeGraphPass):
    """
    Tensor shaping pass common to gpu and cpu.
    Currently used in DotOp.
    """

    @generic_method(dispatch_base_type=Op)
    def visit(self, op, *args):
        pass

    @visit.on_type(DotOp)
    def visit(self, op, x, y):
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
                out = DotLowDimension(x, y, axes=(), bias=op.bias)
            elif len(x.axes) == 1:
                y = Transpose(y)
                out = DotLowDimension(y, x, axes=y.axes[0], bias=op.bias)
            elif len(y.axes) == 1:
                out = DotLowDimension(x, y, axes=x.axes[0], bias=op.bias)
            else:
                out = DotLowDimension(x, y, axes=([op.x_out_axes.flatten(True),
                                                   op.y_out_axes.flatten(True)]), bias=op.bias)
            out = unflatten(out)
            out = ReorderAxes(out, out_axes)

        self.replace_op(op, out)


class CPUTensorShaping(PeepholeGraphPass):
    """TODO."""
    @generic_method(dispatch_base_type=Op)
    def visit(self, op, *args):
        pass

    @visit.on_type(ContiguousOp)
    def visit(self, op, x):
        if x.tensor_description().c_contiguous:
            self.replace_op(op, x)

    @visit.on_type(ReorderAxes)
    def visit(self, op, x):
        if op.axes == x.axes:
            self.replace_op(op, x)

    @visit.on_type(BroadcastOp)
    def visit(self, op, x):
        if op.axes == x.axes:
            self.replace_op(op, x)


class SimplePrune(PeepholeGraphPass):
    """TODO."""
    @generic_method()
    def visit(self, op, *args):
        """
        TODO.

        Arguments:
          op: TODO
        """
        pass

    @visit.on_type(NegativeOp)
    def visit(self, op, x):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
        if x.is_scalar and x.is_constant:
            self.replace_op(op, constant(-x.const))

    @visit.on_type(Multiply)
    def visit(self, op, x, y):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
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
    def visit(self, op, x, y):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
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
    def visit(self, op, x):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
        if x.is_scalar and x.is_constant:
            val = x.const * op.reduction_axes.size
            self.replace_op(op, constant(val))

    @visit.on_type(Prod)
    def visit(self, op, x):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
        if x.is_scalar and x.is_constant:
            val = power(x.const, op.reduction_axes.size)
            self.replace_op(op, constant(val))

    @visit.on_type(LogOp)
    def visit(self, op, x):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
        if isinstance(x, Divide):
            num, denom = self.op_args(x)
            if isinstance(num, ExpOp):
                exp_x, = self.op_args(num)
                self.replace_op(op, exp_x - type(op)(denom))
        elif isinstance(x, ExpOp):
            exp_x, = self.op_args(x)
            self.replace_op(op, exp_x)
