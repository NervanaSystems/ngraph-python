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

from ngraph.transformers.passes.passes import PeepholeGraphPass
from ngraph.util.generics import generic_method
from ngraph.op_graph.pooling import PoolingOp, BpropPoolOp
from ngraph.op_graph.convolution import ConvolutionOp
from ngraph.op_graph.op_graph import BroadcastOp, broadcast, DotOp, ReductionOp, make_axes, \
    axes_with_order, flatten_at, Transpose, unflatten, ReorderAxes, ContiguousOp, \
    OneHotTwoDimOp, BinaryElementWiseAxesOp, AssignOp, DotOneDimensional, DotTwoDimensional, \
    DotTwoByOne, OneHotOp, Flatten, \
    Op, Sum, UnaryElementwiseAxesOp, \
    SetItemOp, tensor_slice
from ngraph.op_graph.axes import make_axis, FlattenedAxis


def _is_strides_contiguous(shape, strides):
    if all(v == 0 for v in strides) or all(v == strides[0] for v in strides):
        return True

    contiguous_strides = [strides[-1]]
    for dim in reversed(shape[1:]):
        contiguous_strides.insert(0, contiguous_strides[0] * dim)
    return (tuple(contiguous_strides) == strides)


class GPUTensorLayout(PeepholeGraphPass):
    """TODO."""
    @generic_method(dispatch_base_type=Op)
    def visit(self, op):
        """
        Base case.
        """
        pass

    @visit.on_type(PoolingOp)
    def visit(self, op):
        """
        Convolution implementation requires contiguous layout.
        """
        inputs = op.args[0]
        if not isinstance(inputs, ContiguousOp):
            new_op = PoolingOp(op.pool_params, ContiguousOp(inputs), axes=op.axes)
            self.replace_op(op, new_op)

    @visit.on_type(BpropPoolOp)
    def visit(self, op):
        """
        Convolution implementation requires contiguous layout.
        """
        deltas = op.args[0]
        if not isinstance(deltas, ContiguousOp):
            new_op = BpropPoolOp(ContiguousOp(deltas), op.inputs, op.fprop)
            self.replace_op(op, new_op)

    @visit.on_type(ConvolutionOp)
    def visit(self, op):
        """
        Convolution implementation requires contiguous layout.
        """
        inputs, filters = op.args

        replace = False
        if not isinstance(inputs, ContiguousOp):
            inputs = ContiguousOp(inputs)
            replace = True

        if not isinstance(filters, ContiguousOp):
            filters = ContiguousOp(filters)
            replace = True

        if replace:
            self.replace_op(op, ConvolutionOp(op.conv_params, inputs, filters, axes=op.axes))

    @visit.on_type(SetItemOp)
    def visit(self, op):
        # PyCuda cannot copy in opposite directions
        tensor = op.args[0]
        value = op.args[1]
        slices = op.item
        new_slices = []
        copy_slices = []
        flip = False
        for s in slices:
            if isinstance(s, slice) and s.step is not None and s.step < 0:
                new_slices.append(slice(s.start, s.stop, -s.step))
                copy_slices.append(slice(None, None, -1))
                flip = True
            elif isinstance(s, slice):
                copy_slices.append(slice(None))
                new_slices.append(s)
            else:
                new_slices.append(s)
        if flip:
            self.replace_op(op, SetItemOp(tensor, new_slices, tensor_slice(value, copy_slices)))


class GPUContiguousPrune(PeepholeGraphPass):
    def match_pattern(self, op, pattern):
        """
        Match a pattern in the graph terminating at the specified op

        Returns: list of ops matching the pattern
        """
        for arg in op.args:
            if type(arg) == pattern[-1]:
                if len(pattern) == 1:
                    return [arg]
                else:
                    match = self.match_pattern(arg, pattern[:-1])
                    if match is not None:
                        return match + [arg]

        return None

    def visit_ew_kernel_op(self, op):
        # Look for contiguous op previous to this op
        match = self.match_pattern(op, (ContiguousOp, Flatten))
        if match is not None:
            td = match[0].args[0].tensor_description()
            if not td.c_contiguous:
                # We can still remove the contig op as long as flattened axes are contig
                can_remove = True
                for axis in match[1].axes:
                    if type(axis) == FlattenedAxis:
                        strides = [td.strides[td.axes.index_unique(a)] for a in axis.axes]
                        shape = [td.strides[td.axes.index_unique(a)] for a in axis.axes]
                        if not _is_strides_contiguous(shape, strides):
                            can_remove = False

                if can_remove:
                    self.replace_op(match[0], match[0].args[0])

    """TODO."""
    @generic_method(dispatch_base_type=Op)
    def visit(self, op):
        """
        Base case.
        """
        pass

    @visit.on_type(UnaryElementwiseAxesOp)
    def visit(self, op):
        self.visit_ew_kernel_op(op)

    @visit.on_type(BinaryElementWiseAxesOp)
    def visit(self, op):
        self.visit_ew_kernel_op(op)

    @visit.on_type(ReductionOp)
    def visit(self, op):
        self.visit_ew_kernel_op(op)

    @visit.on_type(AssignOp)
    def visit(self, op):
        self.visit_ew_kernel_op(op)

    @visit.on_type(ContiguousOp)
    def visit(self, op):
        if op.args[0].tensor_description().c_contiguous:
            self.replace_op(op, op.args[0])


def to_set(axis):
    if type(axis) == list:
        return set(axis)
    else:
        return set([axis])


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


def flatten_op(op, axes, axes_list, reduction_axes=None, axis=None):
    new_order = flatten(axes_list)
    if new_order != range(len(new_order)):
        reordered_axes = make_axes(tuple(flatten(axes)))
        out_args = []
        for arg in op.args:
            out_args.append(Flatten(ReorderAxes(arg, axes=reordered_axes), axes=axes))

        op_type = type(op)
        if reduction_axes is None:
            if axis is None:
                new_op = op_type(*out_args)
            else:
                new_op = op_type(*out_args, axis=axis)
        else:
            new_op = op_type(*out_args, reduction_axes=reduction_axes)

        if type(op) == AssignOp:
            return new_op
        else:
            return ReorderAxes(unflatten(new_op), axes=op.axes)
    else:
        out_args = []
        for arg in op.args:
            out_args.append(Flatten(arg, axes=axes))

        op_type = type(op)
        if reduction_axes is None:
            if axis is None:
                new_op = op_type(*out_args)
            else:
                new_op = op_type(*out_args, axis=axis)
        else:
            new_op = op_type(*out_args, reduction_axes=reduction_axes)

        if type(op) == AssignOp:
            return new_op
        else:
            return unflatten(new_op)


class GPUTensorShaping(PeepholeGraphPass):
    def get_new_axes(self, shape, strides, preserve_axes=None):
        dims = len(shape)
        axes = list(range(dims))
        preserve_set = to_set(preserve_axes)

        def merge_axes(axis1, axis2):
            if type(axis1) == list:
                if type(axis2) == list:
                    return (axis1 + axis2)
                else:
                    return (axis1 + [axis2])
            else:
                if type(axis2) == list:
                    return ([axis1] + axis2)
                else:
                    return [axis1, axis2]

        def can_merge(index1, index2):
            stride1 = strides[index1]
            stride2 = strides[index2]
            shape1 = shape[index1]

            preserve1 = len(preserve_set & to_set(axes[index1])) > 0
            preserve2 = len(preserve_set & to_set(axes[index2])) > 0

            if (stride1 * shape1) == stride2 and ((preserve1 and preserve2) or
                                                  ((not preserve1) and (not preserve2))):
                return True

            return False

        # Merge any axes with same strides
        duplicates = set([x for x in strides if strides.count(x) > 1])
        while len(duplicates) > 0:
            stride = next(iter(duplicates))

            to_merge = []
            for index in range(len(strides)):
                if strides[index] == stride:
                    if type(axes[index]) != int or axes[index] not in preserve_set:
                        to_merge.append(axes[index])

            while len(to_merge) > 1:
                index = axes.index(to_merge.pop(1))

                strides.pop(index)
                shape1 = shape.pop(index)
                axis = axes.pop(index)

                index = axes.index(to_merge[0])
                shape[index] = shape[index] * shape1
                axes[index] = merge_axes(axes[index], axis)
                to_merge[0] = axes[index]

            duplicates.remove(stride)

        # Sort by increasing stride and try to merge each
        sorted_strides = list(strides)
        sorted_strides.sort()
        stride_index = 0
        while len(axes) > 3 and stride_index != (len(axes) - 1):
            index1 = strides.index(sorted_strides[stride_index])
            index2 = strides.index(sorted_strides[stride_index + 1])

            if can_merge(index1, index2):
                if index1 < index2:
                    shape[index1] = shape[index1] * shape[index2]
                    axes[index1] = merge_axes(axes[index1], axes[index2])
                    strides[index1] = strides.pop(index2)
                    shape.pop(index2)
                    axes.pop(index2)
                else:
                    shape[index2] = shape[index1] * shape[index2]
                    axes[index2] = merge_axes(axes[index2], axes[index1])
                    strides[index2] = strides.pop(index1)
                    shape.pop(index1)
                    axes.pop(index1)

                sorted_strides = list(strides)
                sorted_strides.sort()
            else:
                stride_index += 1

        # Check if preserved axes have been merged into a single axis
        out_preserve = None
        if preserve_axes is not None:
            preserve_satisfied = False
            for axis in axes:
                if to_set(axis) == preserve_set:
                    preserve_satisfied = True
                    out_preserve = axis
                    break
        else:
            preserve_satisfied = True

        # Force merging of all preserved axes
        if not preserve_satisfied:
            to_merge = []
            for axis in axes:
                if len(preserve_set & to_set(axis)) > 0:
                    to_merge.append(axis)

            # Merge all axes in list
            merged_axis = []
            for axis in to_merge:
                merged_axis = merge_axes(merged_axis, axis)
                axes.remove(axis)

            if (dims - 1) in merged_axis:
                axes.append(merged_axis)
            else:
                axes.insert(0, merged_axis)

            out_preserve = merged_axis

        # Forced dimshuffle cases
        if len(axes) > 3:
            num_to_merge = len(axes) - 2
            to_merge = []
            for axis in axes:
                if axis != out_preserve:
                    to_merge.append(axis)
                    if len(to_merge) == num_to_merge:
                        break

            merged_axis = []
            for axis in to_merge:
                merged_axis = merge_axes(merged_axis, axis)
                axes.remove(axis)

            axes.insert(0, merged_axis)

        if preserve_axes is not None:
            return (axes, out_preserve)
        else:
            return axes

    @generic_method(dispatch_base_type=Op)
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO
        """
        pass

    @visit.on_type(OneHotTwoDimOp)
    def visit(self, op):
        pass

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

    @visit.on_type(UnaryElementwiseAxesOp)
    def visit(self, op):
        arg0 = op.args[0]
        arg0_td = arg0.tensor_description()

        if len(arg0_td.shape) > 3:
            axes_list = self.get_new_axes(list(arg0_td.shape), list(arg0_td.strides))
            axes = []
            for axis in axes_list:
                if type(axis) == list:
                    to_compound = [arg0_td.axes[a] for a in axis]
                    axes.append(FlattenedAxis((tuple(to_compound))))
                else:
                    axes.append(arg0_td.axes[axis])

            axes = make_axes(tuple(axes))
            new_op = flatten_op(op, axes, axes_list)
            self.replace_op(op, new_op)

    @visit.on_type(BinaryElementWiseAxesOp)
    def visit(self, op):
        arg0 = op.args[0]
        arg0_td = arg0.tensor_description()

        if len(arg0_td.shape) > 3:
            axes_list = self.get_new_axes(list(arg0_td.shape), list(arg0_td.strides))
            axes = []
            for axis in axes_list:
                if type(axis) == list:
                    to_compound = [arg0_td.axes[a] for a in axis]
                    axes.append(FlattenedAxis(tuple(to_compound)))
                else:
                    axes.append(arg0_td.axes[axis])

            axes = make_axes(tuple(axes))
            new_op = flatten_op(op, axes, axes_list)
            self.replace_op(op, new_op)

    @visit.on_type(AssignOp)
    def visit(self, op):
        tensor, val = op.args
        val_td = val.tensor_description()

        if len(val_td.shape) > 3:
            axes_list = self.get_new_axes(list(val_td.shape), list(val_td.strides))
            axes = []
            for axis in axes_list:
                if type(axis) == list:
                    to_compound = [val_td.axes[a] for a in axis]
                    axes.append(FlattenedAxis((tuple(to_compound))))
                else:
                    axes.append(val_td.axes[axis])

            axes = make_axes(tuple(axes))
            new_op = flatten_op(op, axes, axes_list)
            self.replace_op(op, new_op)

    @visit.on_type(ReductionOp)
    def visit(self, op):
        arg0 = op.args[0]
        arg0_td = arg0.tensor_description()

        preserve_axes = [arg0_td.axes.index_unique(axis) for axis in op.reduction_axes]
        if len(arg0_td.shape) > 3 or len(preserve_axes) > 1:
            axes_list, red_axis = self.get_new_axes(list(arg0_td.shape), list(arg0_td.strides),
                                                    preserve_axes)
            axes = []
            for axis in axes_list:
                if type(axis) == list:
                    to_compound = [arg0_td.axes[a] for a in axis]
                    new_axis = FlattenedAxis((tuple(to_compound)))
                else:
                    new_axis = arg0_td.axes[axis]
                axes.append(new_axis)
                if axis == red_axis:
                    reduction_axes = new_axis

            axes = make_axes(tuple(axes))
            new_op = flatten_op(op, axes, axes_list, reduction_axes=reduction_axes)
            self.replace_op(op, new_op)

    @visit.on_type(OneHotOp)
    def visit(self, op):
        op_td = op.tensor_description()

        preserve_axes = [op.axes.index_unique(op.axis)]
        if len(op_td.shape) > 3:
            axes_list, red_axis = self.get_new_axes(list(op_td.shape), list(op_td.strides),
                                                    preserve_axes)
            axes = []
            for axis in axes_list:
                if type(axis) == list:
                    to_compound = [op_td.axes[a] for a in axis]
                    new_axis = FlattenedAxis((tuple(to_compound)))
                else:
                    new_axis = op_td.axes[axis]
                axes.append(new_axis)
                if axis == red_axis:
                    reduction_axes = new_axis

            axes_list.pop(axes.index(reduction_axes))
            axes.remove(reduction_axes)
            axes = make_axes(tuple(axes))
            new_op = flatten_op(op, axes, axes_list, axis=reduction_axes)
            self.replace_op(op, new_op)

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
