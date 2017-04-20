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
from cachetools import cached, LRUCache

from ngraph.op_graph.op_graph import TensorSliceOp, BroadcastOp, ReorderAxes, Flatten, \
    AxesCastOp, TensorValueOp, Unflatten, ExpandDims, SequentialOp, Transpose
from ngraph.op_graph.axes import Axes

from ngraph.transformers.passes.layout import LayoutAssignment, BinaryLayoutConstraint


def flatten(l):
    """
    Flattens a nested list into a single list

    Arguments:
        l: list to flatten

    Returns:
        Flattened list
    """
    out = []
    for item in l:
        if type(item) == list:
            out = out + flatten(item)
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


@cached(cache=LRUCache(maxsize=100))
def enumerate_axis_orders(axes):
    return enumerate_remaining_axes(list(axes))


def split_points_to_groups(split_points, num_axes):
    result = [list(range(s, split_points[i + 1])) for i, s in enumerate(split_points[:-1])]
    result = [list(range(split_points[0]))] + result + [list(range(split_points[-1], num_axes))]
    return result


@cached(cache=LRUCache(maxsize=100))
def get_split_groups(num_axes, num_groups):
    num_splits = num_groups - 1
    split_points = list(reversed([(num_axes - (i + 1)) for i in range(num_splits)]))
    results = []

    for split in tuple(range(num_splits)):
        limit = 0 if split == 0 else split_points[split - 1]

        while split_points[split] > (limit + 1):
            results.append(split_points_to_groups(split_points, num_axes))
            split_points[split] -= 1

    results.append(split_points_to_groups(split_points, num_axes))

    return results


class StridedLayoutAssignment(LayoutAssignment):
    """
    Generic layout descriptor for a strided array. The layout is implemented
    by a list of axis groups where each group of axes must be contiguous in memory.

    Parameters:
        axes: List of ngraph Axis objects specifying axes stored in this layout
        order: List of axis groups

    Example:
        ng_axes = [Axis(C), Axis(H), Axis(W), Axis(N)]
        axes = [[0, 2], [1], [3]]
        This means C and W are flattened into a single contiguous axis in memory and the
        other axes are unconstrained, meaning they may be strided in any way.
    """
    def __init__(self, axes, order=None):
        self.ng_axes = axes
        if order:
            self.axes = order
        else:
            self.axes = list(range(len(axes)))

    def __str__(self):
        out = "("
        for a in self.axes:
            out = out + "["
            for idx in a:
                out = out + str(self.ng_axes[idx].name) + ", "
            out = out + "] "
        out = out + ")"
        return out

    @classmethod
    def generate_ew_layouts(clss, axes, max_out_axes):
        """
        Generates a set of possible layouts for an elementwise operation.

        Arguments:
            axes: List of axes in the output of the operation
            max_out_axes: The maximum number of strided axes supported by
                the kernel for this operation

        Return:
            A list of layout possibilities for this operation
        """
        # Get list of individual axes
        axes_list = Axes.as_flattened_list(axes)

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
                    layouts.append(clss(axes_list, layout_spec))
        else:
            layouts = [clss(axes_list, [])]

        return layouts

    @classmethod
    def generate_default_layout(clss, axes, max_out_axes):
        """
        Generates a default layout assignment for an elementwise operation

        Arguments:
            axes: List of axes in the output of the operation
            max_out_axes: The maximum number of strided axes supported by
                the kernel for this operation

        Return:
            A list containing a single layout assignment
        """
        axes_list = Axes.as_flattened_list(axes)

        # Need to divide op axes into `max_out_axes` sets
        if len(axes_list) > max_out_axes:
            split_points = [(i + 1) for i in range(max_out_axes - 1)]
            layout = split_points_to_groups(split_points, len(axes_list))
        else:
            layout = [[i] for i in range(len(axes_list))]

        return [clss(axes_list, layout)]


class StridedBinaryLayoutConstraint(BinaryLayoutConstraint):
    """
    Provides functionality for layout constraints based on kernels that use strided array address
    calculation. One of the primary functions of this class is to map axes of an argument coming
    into the op back to the buffer underlying that argument. This includes reshape ops between
    the underlying TensorOp and this op such as reordered axes, broadcast axes, and slices.

    Attributes:
        op: Op underlying this constraint
        arg: Argument to the op underlying this constraint
        arg_axes_list: List of axes present in the arg
        mappings: Maps from an axis in arg_axes_list back to an axis in the argument's origin op
        sliced_out: List of axes that are sliced out of the argument between the origin op and
            this op
    """
    def __init__(self, op, arg):
        self.op = op
        self.arg = arg

        # Build mapping of arg axis position to axis position in buffer
        # Arg axes may be re-ordered, cast, broadcast, sliced between the original
        # buffer and the op being used for this constraint
        predecessor_op = arg
        while isinstance(predecessor_op, SequentialOp):
            predecessor_op = predecessor_op.value_tensor

        self.arg_axes_list = Axes.as_flattened_list(arg.axes)
        self.mappings = {}
        self.sliced_out = []
        for i in range(len(self.arg_axes_list)):
            self.mappings[i] = i

        while not (predecessor_op.is_device_op or isinstance(predecessor_op, TensorValueOp)):
            pred_axes = Axes.as_flattened_list(predecessor_op.axes)
            pred_arg_axes = Axes.as_flattened_list(predecessor_op.args[0].axes)
            if isinstance(predecessor_op, (BroadcastOp, ExpandDims)):
                bcast_axes = [pred_axes.index(a) for a in pred_axes if a not in pred_arg_axes]
                bcast_mappings = [a for a in self.mappings if self.mappings[a] in bcast_axes]
                for bcast in bcast_mappings:
                    self.mappings[bcast] = "bcast"
                for a, p in self.mappings.items():
                    if isinstance(p, int):
                        offset = 0
                        for bcast_axis in bcast_axes:
                            if p > bcast_axis:
                                offset += 1
                        self.mappings[a] = p - offset

                for i in range(len(self.sliced_out)):
                    if self.sliced_out[i][0] in bcast_axes:
                        self.sliced_out[i] = (self.sliced_out[i][0], "bcast")
                    else:
                        new_axis_index = pred_arg_axes.index(pred_axes[self.sliced_out[i][0]])
                        self.sliced_out[i] = (new_axis_index, self.sliced_out[i][1])
            elif isinstance(predecessor_op, TensorSliceOp):
                new_indexes = []
                for index, axis in enumerate(pred_axes):
                    new_index = pred_arg_axes.index(axis)
                    if predecessor_op.slices[new_index] != slice(None, None, None):
                        new_index = ("slice", new_index, predecessor_op.slices[new_index])
                    if new_index != index:
                        for a, p in self.mappings.items():
                            if isinstance(p, int) and p == index:
                                new_indexes.append((a, new_index))

                for a, p in new_indexes:
                    self.mappings[a] = p

                # Find any axes that are sliced out and add these to offset calculations
                rem_axes = [pred_arg_axes.index(a) for a in pred_arg_axes if a not in pred_axes]
                for rm_axis in rem_axes:
                    self.sliced_out.append((rm_axis, predecessor_op.slices[rm_axis]))
            elif isinstance(predecessor_op, (ReorderAxes, Transpose)):
                new_indexes = []
                for a, p in self.mappings.items():
                    if isinstance(p, int):
                        new_indexes.append((a, pred_arg_axes.index(pred_axes[p])))
                for a, p in new_indexes:
                    self.mappings[a] = p

                for i in range(len(self.sliced_out)):
                    new_axis_index = pred_arg_axes.index(pred_axes[self.sliced_out[i][0]])
                    self.sliced_out[i] = (new_axis_index, self.sliced_out[i][1])
            elif isinstance(predecessor_op, AxesCastOp):
                pass
            elif isinstance(predecessor_op, Flatten):
                pass
            elif isinstance(predecessor_op, Unflatten):
                pass
            else:
                raise RuntimeError("Confused")
            predecessor_op = predecessor_op.args[0]
            while isinstance(predecessor_op, SequentialOp):
                predecessor_op = predecessor_op.value_tensor

    def get_layout_transform(self, arg_layout, op_layout, arg):
        return arg

    def get_cost(self, arg_layout, op_layout):
        return 0.0

    def map(self, axis):
        """
        Given an axis from the argument, map it back to an index into the list of axes present
        int the origin op.

        Arguments:
            axis: Axis object present in the arg

        Returns:
            Index into origin op axes

        Example:
            AddOp(axes=(N, H, W)) -> ReorderAxes(axes=(H, W, N)) -> BroadcastOp(axes=(C, H, W, N))
            map(C) => "bcast"
            map(H) => 1
            map(W) => 2
            map(N) => 0
        """
        axis_position = -1
        for index, arg_axis in enumerate(self.arg_axes_list):
            if axis == arg_axis:
                axis_position = index
        return self.mappings[axis_position]

    def group_axis_contig(self, arg_mem_order, op_group):
        """
        Given a list of axes in the argument, check if they are contiguous in memory.

        Arguments:
            arg_mem_order: Argument axes order in memory. Integers which index into
                arg_layout.ng_axes
            op_group: List of Axis object to check for contiguity in the argument

        Returns:
            True if axes are in order and contiguous in the argument
        """
        if not op_group:
            return True

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
            if arg_mem_order[i:i + group_len] == arg_group:
                compatible = True
                break
        return compatible

    def group_axis_strided_valid(self, arg_mem_order, op_group):
        """
        Given a list of axes in the argument, check if they can be accessed by a single stride.
        Broadcast axes are allowed only to be grouped with other broadcast axes. Sliced axes
        must be isolated. Regular axes must be contiguous.

        Arguments:
            arg_mem_order: Argument axes order in memory. Integers which index into
                arg_layout.ng_axes
            op_group: List of Axis object to check for contiguity in the argument

        Returns:
            True if axes can be accessed by a single stride
        """
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
            if arg_mem_order[i:i + group_len] == arg_group:
                compatible = True
                break
        return compatible
