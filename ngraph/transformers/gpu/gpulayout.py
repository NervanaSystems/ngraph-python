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
import collections
import numpy as np

from ngraph.op_graph.op_graph import Argmax, Argmin, ContiguousOp, Op, \
    DotLowDimension, Max, Min, OneHotOp, \
    Power, RngOp, Sum, TensorSizeOp, Fill, TensorDescription, \
    Function, AbsoluteOp, Add, AssignOneDOp, AssignOp, CosOp, Divide, Mod, Equal, \
    ExpOp, Greater, GreaterEqual, Less, LessEqual, LogOp, Maximum, Minimum, \
    Multiply, NegativeOp, NotEqual, ReciprocalOp, SignOp, SinOp, SqrtOp, SquareOp, \
    Subtract, TanhOp, SetItemOp, Prod, UnaryElementWiseOp, BinaryElementWiseOp, \
    ReductionOp, DotOp, TensorOp
from ngraph.op_graph.axes import Axis, Axes, FlattenedAxis
from ngraph.transformers.passes.layout import LayoutAssignment, BinaryLayoutConstraint, \
    UnaryLayoutConstraint


class DimshuffleOp(TensorOp):
    """
    Layout transformation op for GPU

    Parameters:
        x (TensorOp): A tensor.
    """

    def __init__(self, x, in_layout, out_layout, **kwargs):
        super(DimshuffleOp, self).__init__(args=(x,), **kwargs)
        self.in_layout = in_layout
        self.out_layout = out_layout


class Memoize:
    def __init__(self, f):
        self.f = f
        self.cache = dict()

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            return self.f(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.f(*args)
            self.cache[args] = value
            return value


def get_axes_list(axes):
    if isinstance(axes, FlattenedAxis):
        return [get_axes_list(a) for a in axes.axes]
    elif isinstance(axes, Axes):
        return [get_axes_list(a) for a in axes]
    elif isinstance(axes, Axis):
        return axes


def flatten(l):
    out = []
    for item in l:
        if type(item) == list:
            out = out + item
        elif type(item) == FlattenedAxis:
            for axis in item.axes:
                out.append(axis)
        else:
            out.append(item)
    return out


def enumerate_remaining_axes(remaining_axes):
    results = []
    if len(remaining_axes) == 1:
        return [[remaining_axes[0]]]

    for axis in remaining_axes:
        other_axes = [a for a in remaining_axes if a != axis]
        sub_results = enumerate_remaining_axes(other_axes)
        for result in sub_results:
            result.insert(0, axis)
            results.append(result)

    return results


@Memoize
def enumerate_axis_orders(axes):
    return enumerate_remaining_axes(list(axes))


def split_points_to_groups(split_points, num_axes):
    result = [list(range(s, split_points[i+1])) for i, s in enumerate(split_points[:-1])]
    result = [list(range(split_points[0]))] + result + [list(range(split_points[-1], num_axes))]
    return result


@Memoize
def get_split_groups(num_axes, num_groups):
    num_splits = num_groups - 1
    split_points = [(i + 1) for i in range(num_splits)]
    results = []

    for split in reversed(range(num_splits)):
        limit = num_axes if split == (num_splits - 1) else split_points[split + 1]

        while split_points[split] < (limit - 1):
            results.append(split_points_to_groups(split_points, num_axes))
            split_points[split] += 1

    results.append(split_points_to_groups(split_points, num_axes))

    return results


class GPULayoutAssignment(LayoutAssignment):
    def __init__(self, axes, order=None):
        self.ng_axes = axes
        if order:
            self.axes = order
        else:
            self.axes = list(range(len(axes)))

    def __str__(self):
        return str(self.axes)

    @staticmethod
    def generate_ew_layouts(axes, max_out_axes):
        # Get list of individual axes
        axes_list = flatten(get_axes_list(axes))

        # Need to divide op axes into `max_out_axes` sets
        if len(axes_list) > max_out_axes:
            groups = get_split_groups(len(axes_list), max_out_axes)
            num_groups = max_out_axes
        else:
            groups = [[i] for i in range(len(axes_list))]
            num_groups = len(axes_list)

        # Find all permutations of these axis groups
        permutations = enumerate_axis_orders(tuple(range(num_groups)))

        if permutations:
            # Create EW layouts
            layouts = []
            for order in permutations:
                layout_spec = [groups[i] for i in order]
                layouts.append(GPULayoutAssignment(axes, layout_spec))
        else:
            layouts = [GPULayoutAssignment(axes, [])]

        return layouts

    @staticmethod
    def generate_default_layout(axes, max_out_axes):
        axes_list = flatten(get_axes_list(axes))

        # Need to divide op axes into `max_out_axes` sets
        if len(axes_list) > max_out_axes:
            split_points = [(i + 1) for i in range(max_out_axes - 1)]
            layout = split_points_to_groups(split_points, len(axes_list))
            num_groups = max_out_axes
        else:
            layout = [[i] for i in range(len(axes_list))]
            num_groups = len(axes_list)

        return [GPULayoutAssignment(axes, layout)]

    @staticmethod
    def factory(op):
        if isinstance(op, AssignOp):
            return GPULayoutAssignment.generate_ew_layouts(op.args[0].axes, 3)
        elif isinstance(op, UnaryElementWiseOp):
            return GPULayoutAssignment.generate_ew_layouts(op.args[0].axes, 3)
        elif isinstance(op, BinaryElementWiseOp):
            return GPULayoutAssignment.generate_ew_layouts(op.args[0].axes, 3)
        elif isinstance(op, ReductionOp):
            # TODO: make sure reduction axes taken care of
            return GPULayoutAssignment.generate_ew_layouts(op.args[0].axes, 3)
        elif isinstance(op, OneHotOp):
            return GPULayoutAssignment.generate_ew_layouts(op.axes, 3)
        elif isinstance(op, TensorSizeOp):
            return GPULayoutAssignment.generate_default_layout(op.axes, 3)
        elif isinstance(op, DotOp):
            return GPULayoutAssignment.generate_default_layout(op.axes, 3)
        else:
            raise ValueError("Layouts not implemented for op type {}".format(op))


class GPUBinaryLayoutConstraint(BinaryLayoutConstraint):
    def __init__(self, op, arg):
        self.op = op
        self.arg = arg

    def get_layout_transform(self, arg_layout, op_layout, arg):
        return arg

    def get_cost(self, arg_layout, op_layout):
        return 0.0

    def broadcast_axes(self, arg_layout, op_layout):
        arg_axis_list = get_axes_list(arg_layout.ng_axes)
        op_axis_list = get_axes_list(op_layout.ng_axes)
        broadcast_axis_list = [op_axis_list.index(a) for a in op_axis_list if a not in arg_axis_list]
        return broadcast_axis_list

    def group_axis_contig(self, arg_mem_order, group):
        compatible = False
        group_len = len(group)
        for i in range(len(arg_mem_order) - group_len + 1):
            if arg_mem_order[i:i+group_len] == group:
                compatible = True
                break
        return compatible

    @staticmethod
    def factory(op, arg):
        if isinstance(op, AssignOp):
            return GPUStridedLayoutConstraint(op, arg)
        elif isinstance(op, UnaryElementWiseOp):
            return GPUStridedLayoutConstraint(op, arg)
        elif isinstance(op, BinaryElementWiseOp):
            return GPUStridedLayoutConstraint(op, arg)
        elif isinstance(op, ReductionOp):
            # TODO: make sure reduction axes taken care of
            return GPUStridedLayoutConstraint(op, arg)
        elif isinstance(op, OneHotOp):
            return GPUStridedLayoutConstraint(op, arg)
        elif isinstance(op, TensorSizeOp):
            return GPUBinaryLayoutConstraint(op, arg)
        elif isinstance(op, DotOp):
            return GPUDotLayoutConstraint(op, arg)
        else:
            raise ValueError("Layouts not implemented for op type {}".format(op))


class GPUStridedLayoutConstraint(GPUBinaryLayoutConstraint):
    def __init__(self, op, arg):
        super(GPUStridedLayoutConstraint, self).__init__(op, arg)

    def needs_transform(self, arg_layout, op_layout):
        # Flattened arg layout axes list used to determine arg contiguity
        arg_mem_order = flatten(arg_layout.axes)
        bcast_axis = self.broadcast_axes(arg_layout, op_layout)

        # Contiguity requirements come from this op's layout groupings
        compatible = True
        for op_axis in op_layout.axes:
            if all(a in bcast_axis for a in op_axis):
                compatible = True
            elif not self.group_axis_contig(arg_mem_order, op_axis):
                compatible = False
                break

        # Check for reduction axes
        if isinstance(self.op, ReductionOp):
            red_axis = [self.arg.axes.index(a) for a in self.op.reduction_axes]
            if not self.group_axis_contig(arg_mem_order, red_axis):
                compatible = False

        # Compute cost as proportional to tensor size if dimshuffle required
        return (not compatible)

    def get_cost(self, arg_layout, op_layout):
        if self.needs_transform(arg_layout, op_layout):
            return 1.0
        else:
            return 0.0

    def get_layout_transform(self, arg_layout, op_layout, arg):
        if self.needs_transform(arg_layout, op_layout):
            if isinstance(self.op, ReductionOp):
                # Dimshuffle to 3d with out axis groups plus reduction group
                reduction_group = [self.arg.axes.index(a) for a in self.op.reduction_axes]
                out_group = [self.arg.axes.index(a) for a in self.op.axes]
                required_layout = GPULayoutAssignment(arg.axes, out_group + reduction_axes)
                return DimshuffleOp(arg, arg_layout, required_layout, axes=arg.axes)
            else:
                # Dimshuffle to 3d with out axis groups
                return DimshuffleOp(arg, arg_layout, op_layout, axes=self.op.axes)
        else:
            return arg


class GPUConvLayoutConstraint(GPUBinaryLayoutConstraint):
    def __init__(self, op, arg):
        super(GPUConvLayoutConstraint, self).__init__(op, arg)

    def needs_transform(self, arg_layout, op_layout):
        arg_mem_order = flatten(arg_layout.axes)
        if arg_mem_order == list(range(5)):
            return False

        return True

    def get_cost(self, arg_layout, op_layout):
        if self.needs_transform(arg_layout, op_layout):
            return 1.0
        else:
            return 0.0

    def get_layout_transform(self, arg_layout, op_layout, arg):
        if self.needs_transform(arg_layout, op_layout):
            order = [list(range(5))]
            required_layout = GPULayoutAssignment(arg.axes, order)
            return DimshuffleOp(arg, arg_layout, required_layout, axes=arg.axes)
        else:
            return arg


class GPUPoolLayoutConstraint(GPUBinaryLayoutConstraint):
    def __init__(self, op, arg):
        super(GPUPoolLayoutConstraint, self).__init__(op, arg)

    def needs_transform(self, arg_layout, op_layout):
        arg_mem_order = flatten(arg_layout.axes)
        if arg_mem_order == list(range(5)):
            return False

        return True

    def get_cost(self, arg_layout, op_layout):
        if self.needs_transform(arg_layout, op_layout):
            return 1.0
        else:
            return 0.0

    def get_layout_transform(self, arg_layout, op_layout, arg):
        if self.needs_transform(arg_layout, op_layout):
            order = [list(range(5))]
            required_layout = GPULayoutAssignment(arg.axes, order)
            return DimshuffleOp(arg, arg_layout, required_layout, axes=arg.axes)
        else:
            return arg


class GPUDotLayoutConstraint(GPUBinaryLayoutConstraint):
    def __init__(self, op, arg):
        super(GPUDotLayoutConstraint, self).__init__(op, arg)

    def needs_transform(self, arg_layout, op_layout):
        arg_mem_order = flatten(arg_layout.axes)
        out_mem_order = flatten(op_layout.axes)
        args = list(self.op.args)

        if self.arg is args[0]:
            reduction_group = [self.arg.axes.index(a) for a in self.op.x_reduction_axes]
        elif self.arg is args[1]:
            reduction_group = [self.arg.axes.index(a) for a in self.op.y_reduction_axes]
        else:
            raise ValueError("Invalid argument for constraint")

        # Each arg must have two contiguous axes where one matches
        # reduction axes and the other matches one of the output axes
        compatible = False
        red_len = len(reduction_group)
        if arg_mem_order[0:red_len] == reduction_group:
            out_axes_group = [self.op.axes.index(self.arg.axes[a]) for a in arg_mem_order[red_len:]]
            if self.group_axis_contig(out_mem_order, out_axes_group):
                compatible = True
        elif arg_mem_order[-red_len:] == reduction_group:
            out_axes_group = [self.op.axes.index(self.arg.axes[a]) for a in arg_mem_order[:-red_len]]
            if self.group_axis_contig(out_mem_order, out_axes_group):
                compatible = True

        return (not compatible)

    def get_cost(self, arg_layout, op_layout):
        if self.needs_transform(arg_layout, op_layout):
            import pdb; pdb.set_trace()
            return 1.0
        else:
            return 0.0

    def get_layout_transform(self, arg_layout, op_layout, arg):
        if self.needs_transform(arg_layout, op_layout):
            if self.arg is args[0]:
                reduction_group = [self.arg.axes.index(a) for a in self.op.x_reduction_axes]
                out_group = [self.arg.axes.index(a) for a in self.op.x_out_axes]
                required_layout = GPULayoutAssignment(arg.axes, out_group + reduction_group)
                return DimshuffleOp(arg, arg_layout, required_layout, axes=arg.axes)
            else:
                reduction_group = [self.arg.axes.index(a) for a in self.op.y_reduction_axes]
                out_group = [self.arg.axes.index(a) for a in self.op.y_out_axes]
                required_layout = GPULayoutAssignment(arg.axes, reduction_group + out_group)
                return DimshuffleOp(arg, arg_layout, required_layout, axes=arg.axes)
        else:
            return arg

class GPUUnaryLayoutConstraint(UnaryLayoutConstraint):
    def __init__(self):
        pass

    def get_cost(self, op_layout):
        return 0.0
