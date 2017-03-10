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
    AbsoluteOp, Add, AssignOneDOp, AssignOp, CosOp, Divide, Mod, Equal, \
    ExpOp, Greater, GreaterEqual, Less, LessEqual, LogOp, Maximum, Minimum, \
    Multiply, NegativeOp, NotEqual, ReciprocalOp, SignOp, SinOp, SqrtOp, SquareOp, \
    Subtract, TanhOp, SetItemOp, Prod, UnaryElementWiseOp, BinaryElementWiseOp, \
    ReductionOp, DotOp, TensorOp, TensorSliceOp, BroadcastOp, ReorderAxes, Flatten, \
    AxesCastOp, ReshapeOp, TensorValueOp, tdcache
from ngraph.op_graph.convolution import ConvolutionOp, update_conv, bprop_conv
from ngraph.op_graph.axes import Axis, Axes, FlattenedAxis
from ngraph.transformers.passes.layout import LayoutAssignment, BinaryLayoutConstraint, \
    UnaryLayoutConstraint


class DimshuffleOp(TensorOp):
    """
    Layout transformation op for GPU

    Parameters:
        x (TensorOp): A tensor.
    """

    def __init__(self, x, in_view, out_view, axis_order, **kwargs):
        super(DimshuffleOp, self).__init__(args=(x,), axes=x.axes, **kwargs)
        #TODO: dtype?
        self.in_view = in_view
        self.out_view = out_view
        self.axis_order = tuple(axis_order)


class GPUReshapeOp(ReshapeOp):
    def __init__(self, x, view, **kwargs):
        super(GPUReshapeOp, self).__init__(x, axes=x.axes, **kwargs)
        self.layout_view = view

    @tdcache()
    def tensor_description(self):
        td = self.args[0].tensor_description().clone()
        if "layout" in self.metadata:
            td.layout = self.metadata["layout"]
        return td


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
        self.shape = None
        self.strides = None

    def __str__(self):
        out = "("
        for a in self.axes:
            out = out + "["
            for idx in a:
                out = out + str(self.ng_axes[idx].name) + ", "
            out = out + "] "
        out = out + ")"
        out = out + "\nshape: {}, strides {}".format(self.shape, self.strides)
        return out

    def set_shape_strides(self):
        if self.axes:
            shape = []
            strides = [1]
            for axis in reversed(self.axes):
                if len(shape) == len(strides):
                    strides.insert(0, strides[0] * shape[0])
                if axis:
                    ax_lens = [self.ng_axes[a].length for a in axis]
                    shape.insert(0, np.prod(ax_lens))
                else:
                    shape.insert(0, 1)
        else:
            shape = []
            strides = []

        self.shape = tuple(shape)
        self.strides = tuple(strides)

    @staticmethod
    def generate_ew_layouts(axes, max_out_axes):
        # Get list of individual axes
        axes_list = flatten(get_axes_list(axes))

        # Need to divide op axes into `max_out_axes` sets
        if len(axes_list) > max_out_axes:
            groups = get_split_groups(len(axes_list), max_out_axes)
            num_groups = max_out_axes
        else:
            groups = [[[i] for i in range(len(axes_list))]]
            num_groups = len(axes_list)

        # Find all permutations of these axis groups
        permutations = enumerate_axis_orders(tuple(range(num_groups)))

        if permutations:
            # Create EW layouts
            layouts = []
            for group in groups:
                for order in permutations:
                    layout_spec = [group[i] for i in order]
                    layouts.append(GPULayoutAssignment(axes_list, layout_spec))
        else:
            layouts = [GPULayoutAssignment(axes_list, [])]

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

        return [GPULayoutAssignment(axes_list, layout)]

    @staticmethod
    def generate_default_dot_layout(op):
        axes_list = flatten(get_axes_list(op.axes))
        rows_axis = [axes_list.index(a) for a in flatten(get_axes_list(op.x_out_axes))]
        cols_axis = [axes_list.index(a) for a in flatten(get_axes_list(op.y_out_axes))]
        # By default allow first argument to be transposed, but not second
        # TODO: this could be bad for perf some heuristic?
        return [GPUDotLayoutAssignment(True, False, axes_list, [rows_axis, cols_axis])]

    @staticmethod
    def factory(op):
        if isinstance(op, AssignOp):
            return GPULayoutAssignment.generate_ew_layouts(op.args[0].axes, 3)
        elif isinstance(op, UnaryElementWiseOp):
            return GPULayoutAssignment.generate_ew_layouts(op.args[0].axes, 3)
        elif isinstance(op, BinaryElementWiseOp):
            return GPULayoutAssignment.generate_ew_layouts(op.args[0].axes, 3)
        elif isinstance(op, ReductionOp):
            return GPULayoutAssignment.generate_ew_layouts(op.axes, 2)
        elif isinstance(op, OneHotOp):
            return GPULayoutAssignment.generate_ew_layouts(op.axes, 3)
        elif isinstance(op, TensorSizeOp):
            return GPULayoutAssignment.generate_default_layout(op.axes, 3)
        elif isinstance(op, Fill):
            return GPULayoutAssignment.generate_default_layout(op.args[0].axes, 3)
        elif isinstance(op, SetItemOp):
            return GPULayoutAssignment.generate_default_layout(op.args[0].axes, 3)
        elif isinstance(op, DotOp):
            return GPULayoutAssignment.generate_default_dot_layout(op)
        elif isinstance(op, ConvolutionOp):
            return GPULayoutAssignment.generate_default_layout(op.axes, 3)
        elif isinstance(op, bprop_conv):
            return GPULayoutAssignment.generate_default_layout(op.axes, 3)
        elif isinstance(op, update_conv):
            return GPULayoutAssignment.generate_default_layout(op.axes, 3)
        elif isinstance(op, TensorValueOp):
            return GPULayoutAssignment.generate_default_layout(op.tensor.axes, 3)
        elif isinstance(op, InitTensorOp):
            return GPULayoutAssignment.generate_default_layout(op.tensor.axes, 3)
        else:
            raise ValueError("Layouts not implemented for op type {}".format(op))


class GPUDotLayoutAssignment(GPULayoutAssignment):
    def __init__(self, A_trans, B_trans, axes, order=None):
        super(GPUDotLayoutAssignment, self).__init__(axes, order)
        self.A_trans = A_trans
        self.B_trans = B_trans
        if A_trans and B_trans:
            raise NotImplementedError("Can't support Dot op tt")


class GPULayoutView(object):
    def __init__(self, shape, strides):
        self.shape = shape
        self.strides = strides

    def __str__(self):
        return "shape: {}, strides {}".format(self.shape, self.strides)


class GPUBinaryLayoutConstraint(BinaryLayoutConstraint):
    def __init__(self, op, arg):
        self.op = op
        self.arg = arg

        # Build mapping of op axis position to arg axis position
        predecessor_op = arg
        self.arg_axes_list = flatten(get_axes_list(arg.axes))
        self.mappings = {}
        for i in range(len(self.arg_axes_list)):
            self.mappings[i] = i

        while not (predecessor_op.is_device_op or isinstance(predecessor_op, TensorValueOp)):
            if isinstance(predecessor_op, BroadcastOp):
                bcast_axes = [predecessor_op.axes.index(a) for a in predecessor_op.axes if a not in predecessor_op.args[0].axes]
                bcast_mappings = [a for a in self.mappings if self.mappings[a] in bcast_axes]
                for bcast in bcast_mappings:
                    self.mappings[bcast] = "bcast"
                for a, p in self.mappings.iteritems():
                    if isinstance(p, int):
                        offset = 0
                        for bcast_axis in bcast_axes:
                            if p > bcast_axis:
                                offset += 1
                        self.mappings[a] = p - offset
            elif isinstance(predecessor_op, TensorSliceOp):
                index = 0
                for index, s in enumerate(predecessor_op.slices):
                    if s == slice(None, None, None):
                        index += 1
                    else:
                        for a, p in self.mappings.iteritems():
                            if p == index:
                                self.mappings[a] = tuple("slice", p, s)
                                break
            elif isinstance(predecessor_op, ReorderAxes):
                new_indexes = []
                for a, p in self.mappings.iteritems():
                    if isinstance(p, int):
                        new_indexes.append(tuple(a, predecessor_op.args[0].axes.index(predecessor_op.axes[p])))
                for a, p in new_indexes:
                    self.mappings[a] = p
            elif isinstance(predecessor_op, AxesCastOp):
                pass
            elif isinstance(predecessor_op, Flatten):
                pass
            else:
                import pdb; pdb.set_trace()
                raise ValueError("Confused")
            predecessor_op = predecessor_op.args[0]

        return

    def get_layout_transform(self, arg_layout, op_layout, arg):
        return arg

    def get_cost(self, arg_layout, op_layout):
        return 0.0

    def map(self, axis):
        axis_position = -1
        for index, arg_axis in enumerate(self.arg_axes_list):
            if axis == arg_axis:
                axis_position = index
        return self.mappings[axis_position]

    def group_axis_contig(self, arg_mem_order, op_group):
        # Convert op group to arg group
        arg_group = [self.map(a) for a in op_group]
        
        # If broadcasts included, not contiguous
        if any(a == "bcast" for a in arg_group) or len(op_group) == 0:
            return False

        # If slices included, not contiguous
        if any(isinstance(a, tuple) for a in arg_group):
            return False

        compatible = False
        group_len = len(arg_group)
        for i in range(len(arg_mem_order) - group_len + 1):
            if arg_mem_order[i:i+group_len] == arg_group:
                compatible = True
                break
        return compatible

    def group_axis_strided_valid(self, arg_mem_order, op_group):
        # Convert op group to arg group
        arg_group = [self.map(a) for a in op_group]
        
        # If broadcasts included, all axes in group must be broadcast
        if any(a == "bcast" for a in arg_group):
            return all(a == "bcast" for a in arg_group)

        # Slices cannot be grouped with other axes
        if any(isinstance(a, tuple) for a in arg_group):
            return len(arg_group) == 1

        compatible = False
        group_len = len(arg_group)
        for i in range(len(arg_mem_order) - group_len + 1):
            if arg_mem_order[i:i+group_len] == arg_group:
                compatible = True
                break
        return compatible

    def contiguous_layout_view(self, out_groups):
        lengths = []
        for group in out_groups:
            if group:
                lengths.append(np.prod([a.length for a in group]))
            else:
                lengths.append(1)

        shape = []
        strides = [1]
        for l in reversed(lengths[1:]):
            shape.insert(0, l)
            strides.insert(0, shape[0] * strides[0])
        shape.insert(0, lengths[0])
        return GPULayoutView(shape, strides)

    def layout_view(self, arg_mem_order, arg_axes, out_groups):
        # Convert op axis groups to arg axis index groups
        shape = []
        axis_groups = []
        for group in out_groups:
            if group:
                shape.append(np.prod([a.length for a in group]))
                axis_groups.append([self.map(a) for a in group])
            else:
                shape.append(1)
                axis_groups.append(["extra"])

        # Compute axis strides for arg
        base_strides = [1]
        arg_axis_strides = [1] * len(arg_axes)
        for i in reversed(arg_mem_order[1:]):
            base_strides.insert(0, arg_axes[i].length * base_strides[0])
        
        for index, i in enumerate(arg_mem_order):
            arg_axis_strides[i] = base_strides[index]

        # Get strides
        strides = []
        for group in axis_groups:
            if group[-1] == "bcast":
                strides.append(0)
            elif group[-1] == "extra":
                strides.append(1)
            elif isinstance(group[0], tuple):
                strides.append(arg_axis_strides[group[0][1]])
            else:
                strides.append(arg_axis_strides[group[-1]])

        return GPULayoutView(shape, strides)

    def get_dimshuffle(self, arg_mem_order, arg_axes, out_groups, arg):
        # Get contiguous un-flattened view of input tensor
        flattened_out_groups = []
        for group in out_groups:
            if group:
                new_groups = [[a] for a in group]
                flattened_out_groups = flattened_out_groups + new_groups
            else:
                flattened_out_groups.append([])
        dimshuffle_in_view = self.layout_view(arg_mem_order, arg_axes, flattened_out_groups)

        # Determine shuffle order
        axis_order = tuple(range(len(flattened_out_groups)))

        # Get contiguous un-flattened view of output tensor
        dimshuffle_out_view = self.contiguous_layout_view(flattened_out_groups)

        # Compute view for the output tensor
        out_shape = []
        out_strides = []
        for group in reversed(out_groups):
            if group:
                length = np.prod([a.length for a in group])
            else:
                length = 1
            if len(out_strides) == 0:
                out_strides.insert(0, 1)
            else:
                out_strides.insert(0, out_shape[0] * out_strides[0])
            out_shape.insert(0, length)
        op_view = GPULayoutView(out_shape, out_strides)

        out = DimshuffleOp(arg, dimshuffle_in_view, dimshuffle_out_view, axis_order)
        out.metadata["layout"] = op_view
        return out

    def get_reshape(self, arg_mem_order, arg_axes, out_groups, arg):
        out_view = self.layout_view(arg_mem_order, arg_axes, out_groups)
        out = GPUReshapeOp(arg, out_view)
        out.metadata["layout"] = out_view
        return out

    @staticmethod
    def factory(op, arg):
        if isinstance(op, AssignOp):
            return GPUStridedLayoutConstraint(op, arg)
        elif isinstance(op, SetItemOp):
            return GPUStridedLayoutConstraint(op, arg)
        elif isinstance(op, UnaryElementWiseOp):
            return GPUStridedLayoutConstraint(op, arg)
        elif isinstance(op, BinaryElementWiseOp):
            return GPUStridedLayoutConstraint(op, arg)
        elif isinstance(op, ReductionOp):
            return GPUStridedLayoutConstraint(op, arg)
        elif isinstance(op, OneHotOp):
            return GPUStridedLayoutConstraint(op, arg)
        elif isinstance(op, TensorSizeOp):
            return GPUBinaryLayoutConstraint(op, arg)
        elif isinstance(op, Fill):
            return GPUBinaryLayoutConstraint(op, arg)
        elif isinstance(op, DotOp):
            return GPUDotLayoutConstraint(op, arg)
        elif isinstance(op, ConvolutionOp):
            return GPUConvLayoutConstraint(op, arg)
        elif isinstance(op, bprop_conv):
            return GPUConvLayoutConstraint(op, arg)
        elif isinstance(op, update_conv):
            return GPUConvLayoutConstraint(op, arg)
        else:
            raise ValueError("Layouts not implemented for op type {}".format(op))


class GPUStridedLayoutConstraint(GPUBinaryLayoutConstraint):
    def __init__(self, op, arg):
        super(GPUStridedLayoutConstraint, self).__init__(op, arg)

    def needs_transform(self, arg_layout, op_layout):
        # Flattened arg layout axes list used to determine arg contiguity
        arg_mem_order = flatten(arg_layout.axes)

        # Contiguity requirements come from this op's layout groupings
        compatible = True
        for op_axis in op_layout.axes:
            arg_axis = [op_layout.ng_axes[i] for i in op_axis]
            if isinstance(self.op, OneHotOp) and arg_axis[0] == self.op.axis:
                continue
            if not self.group_axis_strided_valid(arg_mem_order, arg_axis):
                compatible = False
                break

        # Check for reduction axes
        if isinstance(self.op, ReductionOp):
            red_axis = [a for a in self.op.reduction_axes]
            if not self.group_axis_strided_valid(arg_mem_order, red_axis):
                compatible = False

        # Compute cost as proportional to tensor size if dimshuffle required
        return (not compatible)

    def get_cost(self, arg_layout, op_layout):
        if self.needs_transform(arg_layout, op_layout):
            return 1.0
        else:
            return 0.0

    def get_layout_transform(self, arg_layout, op_layout, arg):
        # Flattened arg layout axes list used to determine arg contiguity
        arg_mem_order = flatten(arg_layout.axes)
        arg_axes = arg_layout.ng_axes

        if self.needs_transform(arg_layout, op_layout):
            if isinstance(self.op, ReductionOp):
                # Dimshuffle to 3d with out axis groups plus reduction group
                out_groups = [[a for a in self.op.reduction_axes]]
                for op_axis in op_layout.axes:
                    out_groups.append([op_layout.ng_axes[i] for i in op_axis])
                return self.get_dimshuffle(arg_mem_order, arg_axes, out_groups, arg)
            elif isinstance(self.op, OneHotOp):
                # Dimshuffle to 3d with out axis groups other than onehot axis
                out_groups = []
                for op_axis in op_layout.axes:
                    group = [op_layout.ng_axes[i] for i in op_axis]
                    if self.op.axis in group:
                        assert len(group) == 1
                        continue
                    out_groups.append(group)
                return self.get_dimshuffle(arg_mem_order, arg_axes, out_groups, arg)
            else:
                # Dimshuffle to 3d with out axis groups
                out_groups = []
                for op_axis in op_layout.axes:
                    out_groups.append([op_layout.ng_axes[i] for i in op_axis])
                return self.get_dimshuffle(arg_mem_order, arg_axes, out_groups, arg)
        else:
            # Compute derived layout for arg
            if isinstance(self.op, ReductionOp):
                out_groups = [[a for a in self.op.reduction_axes]]
                for op_axis in op_layout.axes:
                    out_groups.append([op_layout.ng_axes[i] for i in op_axis])
                return self.get_reshape(arg_mem_order, arg_axes, out_groups, arg)
            elif isinstance(self.op, OneHotOp):
                out_groups = []
                for op_axis in op_layout.axes:
                    group = [op_layout.ng_axes[i] for i in op_axis]
                    if self.op.axis in group:
                        assert len(group) == 1
                        continue
                    out_groups.append(group)
                return self.get_reshape(arg_mem_order, arg_axes, out_groups, arg)
            else:
                out_groups = []
                for op_axis in op_layout.axes:
                    out_groups.append([op_layout.ng_axes[i] for i in op_axis])
                return self.get_reshape(arg_mem_order, arg_axes, out_groups, arg)

            return arg


class GPUConvLayoutConstraint(GPUBinaryLayoutConstraint):
    def __init__(self, op, arg):
        super(GPUConvLayoutConstraint, self).__init__(op, arg)
        self.order = [flatten(get_axes_list(arg))]

    def needs_transform(self, arg_layout, op_layout):
        arg_mem_order = flatten(arg_layout.axes)
        if not self.group_axis_contig(arg_mem_order, self.order):
            return True

        return False

    def get_cost(self, arg_layout, op_layout):
        if self.needs_transform(arg_layout, op_layout):
            return 1.0
        else:
            return 0.0

    def get_layout_transform(self, arg_layout, op_layout, arg):
        arg_mem_order = flatten(arg_layout.axes)
        arg_axes = arg.axes

        if self.needs_transform(arg_layout, op_layout):
            out_groups = [[a] for a in self.order]
            return self.get_dimshuffle(arg_mem_order, arg_axes, out_groups, arg)
        else:
            return arg


class GPUPoolLayoutConstraint(GPUBinaryLayoutConstraint):
    def __init__(self, op, arg):
        super(GPUPoolLayoutConstraint, self).__init__(op, arg)
        self.order = flatten(get_axes_list(arg))

    def needs_transform(self, arg_layout, op_layout):
        arg_mem_order = flatten(arg_layout.axes)
        if not self.group_axis_contig(arg_mem_order, self.order):
            return True

        return False

    def get_cost(self, arg_layout, op_layout):
        if self.needs_transform(arg_layout, op_layout):
            return 1.0
        else:
            return 0.0

    def get_layout_transform(self, arg_layout, op_layout, arg):
        arg_mem_order = flatten(arg_layout.axes)
        arg_axes = arg_layout.ng_axes

        if self.needs_transform(arg_layout, op_layout):
            out_groups = [[a] for a in self.order]
            return self.get_dimshuffle(arg_mem_order, arg_axes, out_groups, arg)
        else:
            return arg


class GPUDotLayoutConstraint(GPUBinaryLayoutConstraint):
    def __init__(self, op, arg):
        super(GPUDotLayoutConstraint, self).__init__(op, arg)

        args = list(self.op.args)
        self.op_axes = flatten(get_axes_list(self.op.axes))
        if self.arg.forwarded is args[0].forwarded:
            self.operand = 'A'
            self.reduction_axes = flatten(get_axes_list(self.op.x_reduction_axes))
            self.out_axes = flatten(get_axes_list(self.op.x_out_axes))
        elif self.arg.forwarded is args[1].forwarded:
            self.operand = 'B'
            self.reduction_axes = flatten(get_axes_list(self.op.y_reduction_axes))
            self.out_axes = flatten(get_axes_list(self.op.y_out_axes))
        else:
            import pdb; pdb.set_trace()
            raise ValueError("Invalid argument for constraint")

    def needs_transform(self, arg_layout, op_layout):
        arg_mem_order = flatten(arg_layout.axes)
        out_mem_order = flatten(op_layout.axes)

        reduction_group = [a for a in self.reduction_axes]
        out_group = [self.op_axes[i] for i in out_mem_order if self.op_axes[i] in self.out_axes]

        # Check if this argument can be transposed
        if self.operand == 'A':
            can_trans = op_layout.A_trans
        elif self.operand == 'B':
            can_trans = op_layout.B_trans

        # Each arg must have two contiguous axes where one matches
        # reduction axes and the other matches one of the output axes
        if len(reduction_group) == 0 or self.group_axis_contig(arg_mem_order, reduction_group):
            if can_trans:
                if self.group_axis_contig(arg_mem_order, out_group):
                    return False
            else:
                # Make sure operand is not transposed
                if self.operand == 'A':
                    required_layout = out_group + reduction_group
                elif self.operand == 'B':
                    required_layout = reduction_group + out_group

                if self.group_axis_contig(arg_mem_order, required_layout):
                    return False
        
        return True

    def get_cost(self, arg_layout, op_layout):
        if self.needs_transform(arg_layout, op_layout):
            return 1.0
        else:
            return 0.0

    def get_layout_transform(self, arg_layout, op_layout, arg):
        arg_mem_order = flatten(arg_layout.axes)
        arg_axes = arg_layout.ng_axes
        args = list(self.op.args)

        reduction_group = [a for a in self.reduction_axes]
        out_group = [a for a in self.out_axes]

        if self.needs_transform(arg_layout, op_layout):
            if self.arg.forwarded is args[0].forwarded:
                out_groups = [out_group, reduction_group]
                return self.get_dimshuffle(arg_mem_order, arg_axes, out_groups, arg)
            else:
                out_groups = [reduction_group, out_group]
                return self.get_dimshuffle(arg_mem_order, arg_axes, out_groups, arg)
        else:
            if self.arg.forwarded is args[0].forwarded:
                out_groups = [out_group, reduction_group]
                return self.get_reshape(arg_mem_order, arg_axes, out_groups, arg)
            else:
                out_groups = [reduction_group, out_group]
                return self.get_reshape(arg_mem_order, arg_axes, out_groups, arg)

class GPUUnaryLayoutConstraint(UnaryLayoutConstraint):
    def __init__(self):
        pass

    def get_cost(self, op_layout):
        return 0.0
