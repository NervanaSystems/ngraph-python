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
import numpy as np

from ngraph.op_graph.op_graph import OneHotOp, RngOp, TensorSizeOp, Fill, AssignOp, \
    SetItemOp, UnaryElementWiseOp, BinaryElementWiseOp, ReductionOp, DotOp, TensorOp, \
    ReshapeOp, TensorValueOp, AssignableTensorOp, tdcache
from ngraph.op_graph.convolution import ConvolutionOp, update_conv, bprop_conv
from ngraph.op_graph.pooling import PoolingOp, BpropPoolOp
from ngraph.op_graph.axes import Axes
from ngraph.op_graph.lookuptable import LookupTableOp, update_lut, bprop_lut
from ngraph.op_graph.comm_nodes import GPUQueueSendOp, GPUQueueRecvOp
from ngraph.op_graph.ctc import CTCOp

from ngraph.transformers.passes.layout import UnaryLayoutConstraint
from ngraph.transformers.util.layout_common import StridedLayoutAssignment, \
    StridedBinaryLayoutConstraint, flatten


class GPULayoutView(object):
    """
    Contains information needed to create a tensor view for the GPU

    Attributes:
        shape: tuple of integer dimension lengths
        strides: tuple of integer item strides
        offset: item offset from base address of GPU memory allocation
    """
    def __init__(self, shape, strides, offset=0):
        self.shape = tuple([int(i) for i in shape])
        self.strides = tuple([int(i) for i in strides])
        self.offset = int(offset)

    def __str__(self):
        out = "shape: {}, strides {}, offset {}".format(self.shape, self.strides, self.offset)
        return out


class DimshuffleOp(TensorOp):
    """
    Layout transformation op for GPU which does a copy

    Parameters:
        x (TensorOp): A tensor
        in_view (GPULayoutView): View to use for reading the input
        out_view (GPULayoutView): View to use for writing the output
        axis_order (list): List of input axis indexes specifying order to permute
            input axes for the output
    """

    def __init__(self, x, in_view, out_view, axis_order, **kwargs):
        super(DimshuffleOp, self).__init__(args=(x,), axes=x.axes, **kwargs)
        # TODO: dtype?
        self.in_view = in_view
        self.out_view = out_view
        self.axis_order = tuple(axis_order)


class GPUReshapeOp(ReshapeOp):
    """
    Layout transformation op for a GPU which does not copy, but changes shape and/or strides

    Parameters:
        x (TensorOp): A tensor
        view (GPULayoutView): New view to use for reading the tensor
    """
    def __init__(self, x, view, **kwargs):
        super(GPUReshapeOp, self).__init__(x, axes=x.axes, **kwargs)
        self.layout_view = view

    @tdcache()
    def tensor_description(self):
        td = self.args[0].tensor_description().clone()
        if "layout" in self.metadata:
            td.layout = self.metadata["layout"]
        return td


class GPULayoutAssignment(StridedLayoutAssignment):
    """
    GPU implementation of device specific layout descriptor.

    Parameters:
        axes: List of ngraph Axis objects specifying axes stored in this layout
        order: List of axis groups. See definition of StridedLayoutAssignment for
            an example

    Attributes:
        ng_axes: List of Axis objects stored by this layout (unflattened)
        axes: List of lists where each list is a group of axis indices.
        shape: Tensor shape calculated from layout axes groups
        strides: Tensor strides calculated from layout axes groups
        offset: Tensor offset from buffer base
    """
    def __init__(self, axes, order=None):
        super(GPULayoutAssignment, self).__init__(axes, order)
        self.shape = None
        self.strides = None
        self.offset = None

    def __str__(self):
        out = super(GPULayoutAssignment, self).__str__()
        out = out + "\nshape: {}, strides {}".format(self.shape, self.strides)
        return out

    def set_shape_strides(self):
        """
        Compute shape and strides based on axis groups specified by this layout.
        Groups will be stored in the order that they are specified in the layout.
        """
        if self.axes:
            shape = []
            strides = [1]
            for axis in reversed(self.axes):
                if len(shape) == len(strides):
                    strides.insert(0, strides[0] * shape[0])
                if axis:
                    ax_lens = [self.ng_axes[a].length for a in axis]
                    shape.insert(0, int(np.prod(ax_lens)))
                else:
                    shape.insert(0, 1)
        else:
            shape = []
            strides = []

        self.offset = 0
        self.shape = tuple(shape)
        self.strides = tuple(strides)

    @staticmethod
    def generate_default_dot_layout(op):
        """
        Generates the default layout assignment for a dot operation on GPU. By default
        we allow the first operand to be transposed but not the second.

        Output layout of the Dot operation is defined by rows which are the non-reduction
        axes from the first operand and columns which are the non-reduction axes from the
        second operand.

        Arguments:
            op (DotOp): op to generate layout for

        Returns:
            GPUDotLayoutAssignment for this op
        """
        axes_list = Axes.as_flattened_list(op.axes)
        rows_axis = [axes_list.index(a) for a in Axes.as_flattened_list(op.x_out_axes)]
        cols_axis = [axes_list.index(a) for a in Axes.as_flattened_list(op.y_out_axes)]
        # By default allow first argument to be transposed, but not second
        # TODO: this could be bad for perf some heuristic?
        return [GPUDotLayoutAssignment(True, False, axes_list, [rows_axis, cols_axis])]

    @staticmethod
    def generate_default_onehot_layout(op):
        """
        Generates the default layout assignment for a onehot operation on GPU.

        Arguments:
            op (OneHotOp): op to generate layout for

        Returns:
            GPULayoutAssignment for this op
        """
        axes_list = Axes.as_flattened_list(op.axes)
        oh_axis = axes_list.index(op.axis)
        other_group = [i for i, a in enumerate(axes_list) if a is not op.axis]

        if oh_axis == 0:
            return [GPUDotLayoutAssignment(True, False, axes_list, [[oh_axis], other_group])]
        elif oh_axis == (len(axes_list) - 1):
            return [GPUDotLayoutAssignment(True, False, axes_list, [other_group, [oh_axis]])]
        else:
            group0 = [i for i in other_group if i < oh_axis]
            group1 = [i for i in other_group if i > oh_axis]
            return [GPUDotLayoutAssignment(True, False, axes_list, [group0, [oh_axis], group1])]

    @staticmethod
    def generate_default_lut_layout(op):
        """
        Generates the default layout assignment for a lookup table operation on GPU.

        Arguments:
            op (LookupTableOp): op to generate layout for

        Returns:
            GPULayoutAssignment for this op
        """
        axes_list = Axes.as_flattened_list(op.axes)
        groups = Axes.as_nested_list(op.axes)
        layout = []
        for group in groups:
            if isinstance(group, list):
                layout.append([axes_list.index(a) for a in group])
            else:
                layout.append([axes_list.index(group)])
        return [GPULayoutAssignment(axes_list, layout)]


class GPUDotLayoutAssignment(GPULayoutAssignment):
    """
    Dot operation on GPU supports one operand being transposed, but not both. This
    layout assignment adds a parameter that specifies which operand may be transposed.

    Attributes:
        A_trans: True if the first operand can be transposed
        B_trans: True if the second operand can be transposed
    """
    def __init__(self, A_trans, B_trans, axes, order=None):
        super(GPUDotLayoutAssignment, self).__init__(axes, order)
        self.A_trans = A_trans
        self.B_trans = B_trans
        if A_trans and B_trans:
            raise NotImplementedError("Can't support Dot op tt")


class GPUBinaryLayoutConstraint(StridedBinaryLayoutConstraint):
    """
    Base class for GPU binary layout constraints. Can generate dimshuffle and reshape
    ops based on constraint definition and two op layout assignments.
    """
    def __init__(self, op, arg):
        super(GPUBinaryLayoutConstraint, self).__init__(op, arg)

    def contiguous_layout_view(self, out_groups):
        """
        Generates a contiguous view for the axes specified by out_groups. Will place
        groups in memory in the order they are specified in the argument.

        Arguments:
            out_groups: List of axis groups, where each group is a list of Axis objects
                which must be contiguous.

        Returns:
            Tensor view where all groups are contiguous in memory
        """
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
        """
        Given an input tensor, generate a view of that tensor with specified ordered
        axes. Axis groups must be contiguous in the argument; this condition is not checked
        by this method.

        Arguments:
            arg_mem_order: List of axis indexes specifying order in memory
            arg_axes: List of Axis objects indexed by arg_mem_order
            out_groups: List of Axis groups specifying desired view

        Returns:
            GPULayoutView which maps the argument tensor to the view specified by out_groups
        """
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
        offset = 0
        for axis_index, group in enumerate(axis_groups):
            if group[-1] == "bcast":
                strides.append(0)
            elif group[-1] == "extra":
                strides.append(1)
            elif isinstance(group[0], tuple):
                arg_mem_axis = group[0][1]
                if isinstance(group[0][2], slice):
                    slice_stride = group[0][2].step
                    if slice_stride is None:
                        slice_stride = 1
                    start = group[0][2].start
                    if start is None:
                        if slice_stride < 0:
                            start = arg_axes[arg_mem_order[arg_mem_axis]].length - 1
                        else:
                            start = 0
                else:
                    start = group[0][2]
                    slice_stride = 1

                strides.append(arg_axis_strides[arg_mem_axis] * slice_stride)
                # Add slice to offset
                offset += (start * arg_axis_strides[arg_mem_axis])
            else:
                strides.append(arg_axis_strides[group[-1]])

        # Add any offset from axes that are sliced out of the view
        for axis, s in self.sliced_out:
            if s != "bcast":
                stride = arg_axis_strides[axis]
                if isinstance(s, slice):
                    offset += (s.start * stride)
                else:
                    offset += (s * stride)

        return GPULayoutView(shape, strides, offset=offset)

    def get_dimshuffle(self, arg_mem_order, arg_axes, out_groups, arg):
        """
        Given an input tensor, generate a dimshuffle operation which will output a tensor
        compatible with the view specified by out_groups.

        Arguments:
            arg_mem_order: List of axis indexes specifying order in memory
            arg_axes: List of Axis objects indexed by arg_mem_order
            out_groups: List of Axis groups specifying desired view
            arg: The op producing this argument

        Returns:
            DimshuffleOp which converts the argument to a tensor with the desired axes groups
        """
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
        """
        Given an input tensor, generate a ReshapeOp that produces a view of the original tensor
        which is defined by out_groups. Axis groups must be contiguous in the argument; this
        condition is not checked by this method.

        Arguments:
            arg_mem_order: List of axis indexes specifying order in memory
            arg_axes: List of Axis objects indexed by arg_mem_order
            out_groups: List of Axis groups specifying desired view
            arg: The op producing this argument

        Returns:
            GPUReshapeOp which contains a view of the original tensor with the desired axes groups
        """
        out_view = self.layout_view(arg_mem_order, arg_axes, out_groups)
        out = GPUReshapeOp(arg, out_view)
        out.metadata["layout"] = out_view
        return out

    def needs_transform(self, arg_layout, op_layout):
        """
        Default constraint requires no dimshuffle

        Arguments:
            arg_layout (GPULayoutAssignment): layout of the argument
            op_layout: (GPULayoutAssignment): layout required by the op

        Returns:
            True if a DimshuffleOp is needed to convert the arg
        """
        return False

    def get_cost(self, arg_layout, op_layout):
        """
        Returns cost of this constraint given the pair of layouts. If no DimshuffleOp is
        needed, the cost is 0. Otherwise the cost is non-zero.
        TODO: model of dimshuffle cost

        Arguments:
            arg_layout (GPULayoutAssignment): layout of the argument
            op_layout: (GPULayoutAssignment): layout required by the op

        Returns:
            Cost of the constraint
        """
        if self.needs_transform(arg_layout, op_layout):
            return 1.0
        else:
            return 0.0


class GPUEWLayoutConstraint(GPUBinaryLayoutConstraint):
    """
    GPU layout constraint for elementwise operations and most CUDA C kernels. The op layout
    specifies how the kernel will read a tensor with strides and shape. The op layout also
    specifies the storage order for the axes in the output. This constraint checks if an argument
    with a specified layout can be read by a kernel with a specified op layout.

    Attributes:
        red_axes: Axes which are reduced by the op and are therefore present in the input but not
            in the output.
    """
    def __init__(self, op, arg):
        super(GPUEWLayoutConstraint, self).__init__(op, arg)
        if isinstance(op, ReductionOp):
            self.red_axes = Axes.as_flattened_list(op.reduction_axes)
        else:
            self.red_axes = None

    def needs_transform(self, arg_layout, op_layout):
        """
        Given the op layout and argument layout, check if a DimshuffleOp is needed to convert
        the argument to a suitable layout.

        Arguments:
            arg_layout (GPULayoutAssignment): layout of the argument
            op_layout: (GPULayoutAssignment): layout required by the op

        Returns:
            True if a DimshuffleOp is needed to convert the arg
        """
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
            red_axis = [a for a in self.red_axes]
            if not self.group_axis_strided_valid(arg_mem_order, red_axis):
                compatible = False

        return (not compatible)

    def get_layout_transform(self, arg_layout, op_layout, arg):
        """
        Given the op layout and argument layout, check if a DimshuffleOp is needed to convert
        the argument to a suitable layout. Generates either a DimshuffleOp or GPUReshapeOp for
        the argument which produces a view which satisfies the op_layout assignment.

        Arguments:
            arg_layout (GPULayoutAssignment): layout of the argument
            op_layout: (GPULayoutAssignment): layout required by the op
            arg (TensorOp): Op producing the argument

        Returns:
            Either a GPUReshapeOp if no transform is needed, or a DimshuffleOp which satisfies
            the requirements of the op_layout
        """
        # Flattened arg layout axes list used to determine arg contiguity
        arg_mem_order = flatten(arg_layout.axes)
        arg_axes = arg_layout.ng_axes

        if self.needs_transform(arg_layout, op_layout):
            if isinstance(self.op, ReductionOp):
                # Dimshuffle to 3d with out axis groups plus reduction group
                out_groups = [[a for a in self.red_axes]] if self.red_axes else []
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
                out_groups = [[a for a in self.red_axes]] if self.red_axes else []
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


class GPUFixedLayoutConstraint(GPUBinaryLayoutConstraint):
    """
    GPU layout constraint for an operation which only supports fully contiguous arguments. An
    example is convolution which requires the axes to be contiguous and match the order supported
    by the kernel.

    Attributes:
        order: List of axes in order which must be satisfied by the argument
    """
    def __init__(self, op, arg, axes):
        super(GPUFixedLayoutConstraint, self).__init__(op, arg)
        self.order = Axes.as_flattened_list(axes)

    def needs_transform(self, arg_layout, op_layout):
        """
        Checks if all axes in self.order are contiguous in the argument.

        Arguments:
            arg_layout (GPULayoutAssignment): layout of the argument
            op_layout: (GPULayoutAssignment): layout required by the op

        Returns:
            True if a DimshuffleOp is needed to convert the arg
        """
        arg_mem_order = flatten(arg_layout.axes)
        if not self.group_axis_contig(arg_mem_order, self.order):
            return True

        return False

    def get_layout_transform(self, arg_layout, op_layout, arg):
        """
        Generates either a DimshuffleOp or GPUReshapeOp for the argument that produces a view
        which satisfies contiguous order requirement.

        Arguments:
            arg_layout (GPULayoutAssignment): layout of the argument
            op_layout: (GPULayoutAssignment): layout required by the op
            arg (TensorOp): Op producing the argument

        Either a GPUReshapeOp if no transform is needed, or a DimshuffleOp which satisfies
            the requirements of the op_layout
        """
        arg_mem_order = flatten(arg_layout.axes)
        arg_axes = arg_layout.ng_axes

        if self.needs_transform(arg_layout, op_layout):
            return self.get_dimshuffle(arg_mem_order, arg_axes, [self.order], arg)
        else:
            return self.get_reshape(arg_mem_order, arg_axes, [self.order], arg)


class GPUDotLayoutConstraint(GPUBinaryLayoutConstraint):
    """
    GPU layout constraint for dot operation. Requires the reduction axes in the argument
    to be contiguous in the same order as the op specifies. Also requires the non-reduction
    axes to be contiguous.

    Attributes:
        op_axes: list of axes in the output
        operand: Specifies if this is the left or right operand
        reduction_axes: List of axes used for the inner product
        out_axes: List of axes which are present in the argument and op output
    """
    def __init__(self, op, arg):
        super(GPUDotLayoutConstraint, self).__init__(op, arg)

        args = list(self.op.args)
        self.op_axes = Axes.as_flattened_list(self.op.axes)
        if self.arg.forwarded is args[0].forwarded:
            self.operand = 'A'
            self.reduction_axes = Axes.as_flattened_list(self.op.reduction_axes)
            self.out_axes = Axes.as_flattened_list(self.op.x_out_axes)
        elif self.arg.forwarded is args[1].forwarded:
            self.operand = 'B'
            self.reduction_axes = Axes.as_flattened_list(self.op.reduction_axes)
            self.out_axes = Axes.as_flattened_list(self.op.y_out_axes)
        else:
            raise ValueError("Invalid argument for constraint")

    def needs_transform(self, arg_layout, op_layout):
        """
        Checks if reduction_axes and out_axes are contiguous and if the argument
        meets the transpose requirements of the op

        Arguments:
            arg_layout (GPULayoutAssignment): layout of the argument
            op_layout: (GPULayoutAssignment): layout required by the op

        Returns:
            True if a DimshuffleOp is needed to convert the arg
        """
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

    def get_layout_transform(self, arg_layout, op_layout, arg):
        """
        Generates either a DimshuffleOp or GPUReshapeOp for the argument that produces a view
        which satisfies the dot op layout.

        Arguments:
            arg_layout (GPULayoutAssignment): layout of the argument
            op_layout: (GPULayoutAssignment): layout required by the op
            arg (TensorOp): Op producing the argument

        Either a GPUReshapeOp if no transform is needed, or a DimshuffleOp which satisfies
            the requirements of the op_layout
        """
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


class GPUSetItemLayoutConstraint(GPUBinaryLayoutConstraint):
    """
    Simple constraint for SetItemOp, which can handle any number of strided axes.
    """
    def __init__(self, op, arg):
        super(GPUSetItemLayoutConstraint, self).__init__(op, arg)

    def get_cost(self, arg_layout, op_layout):
        return 0.0

    def get_layout_transform(self, arg_layout, op_layout, arg):
        """
        Returns a reshape view of the argument strided to match the SetItemOp axes

        Arguments:
            arg_layout (GPULayoutAssignment): layout of the argument
            op_layout: (GPULayoutAssignment): layout required by the op
            arg (TensorOp): Op producing the argument

        A GPUReshapeOp which satisfies the requirements of the op_layout
        """
        arg_mem_order = flatten(arg_layout.axes)
        arg_view_axes = Axes.as_flattened_list(arg.axes)
        arg_axes = arg_layout.ng_axes
        out_groups = [[a] for a in arg_view_axes]
        return self.get_reshape(arg_mem_order, arg_axes, out_groups, arg)


class GPULutLayoutConstraint(GPUBinaryLayoutConstraint):
    """
    Constraint for LookupTableOp. The lookuptable must have two contiguous axes
    and the index list must have one contiguous axis.

    Attributes:
        order: List of axis groups which must be contiguous
    """
    def __init__(self, op, arg):
        super(GPULutLayoutConstraint, self).__init__(op, arg)
        if len(arg.axes) == 2:
            self.order = [Axes.as_flattened_list(Axes(arg.axes[0])),
                          Axes.as_flattened_list(Axes(arg.axes[1]))]
        else:
            self.order = [Axes.as_flattened_list(arg.axes)]

    def needs_transform(self, arg_layout, op_layout):
        """
        Checks if all axes in self.order are contiguous in the argument.

        Arguments:
            arg_layout (GPULayoutAssignment): layout of the argument
            op_layout: (GPULayoutAssignment): layout required by the op

        Returns:
            True if a DimshuffleOp is needed to convert the arg
        """
        arg_mem_order = flatten(arg_layout.axes)
        for group in self.order:
            if not self.group_axis_contig(arg_mem_order, group):
                return True

        return False

    def get_layout_transform(self, arg_layout, op_layout, arg):
        arg_mem_order = flatten(arg_layout.axes)
        arg_axes = arg_layout.ng_axes

        if self.needs_transform(arg_layout, op_layout):
            return self.get_dimshuffle(arg_mem_order, arg_axes, self.order, arg)
        else:
            return self.get_reshape(arg_mem_order, arg_axes, self.order, arg)


class GPUUnaryLayoutConstraint(UnaryLayoutConstraint):
    """
    Placehold for unary constraint
    TODO: This should return a cost for an op based on the layout. Layouts that
    are less optimal for the kernel/implementation should cause this constraint to
    increase in cost.
    """
    def __init__(self):
        pass

    def get_cost(self, op_layout):
        return 0.0


def gpu_layout_factory(op):
    """
    Generates a list of possible layouts given an op

    Arguments:
        op: Computation graph op which runs on the device

    Returns:
        List of possible layout assignment descriptors
    """
    if isinstance(op, AssignOp):
        return GPULayoutAssignment.generate_ew_layouts(op.args[0].axes, 3)
    elif isinstance(op, UnaryElementWiseOp):
        return GPULayoutAssignment.generate_ew_layouts(op.args[0].axes, 3)
    elif isinstance(op, BinaryElementWiseOp):
        return GPULayoutAssignment.generate_ew_layouts(op.args[0].axes, 3)
    elif isinstance(op, ReductionOp):
        return GPULayoutAssignment.generate_ew_layouts(op.axes, 2)
    elif isinstance(op, OneHotOp):
        return GPULayoutAssignment.generate_default_onehot_layout(op)
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
    elif isinstance(op, PoolingOp):
        return GPULayoutAssignment.generate_default_layout(op.axes, 3)
    elif isinstance(op, BpropPoolOp):
        return GPULayoutAssignment.generate_default_layout(op.axes, 3)
    elif isinstance(op, TensorValueOp):
        return GPULayoutAssignment.generate_default_layout(op.tensor.axes, 3)
    elif isinstance(op, AssignableTensorOp):
        return GPULayoutAssignment.generate_default_layout(op.axes, 3)
    elif isinstance(op, LookupTableOp):
        return GPULayoutAssignment.generate_default_lut_layout(op)
    elif isinstance(op, (update_lut, bprop_lut)):
        return GPULayoutAssignment.generate_default_layout(op.axes, 3)
    elif isinstance(op, RngOp):
        return GPULayoutAssignment.generate_default_layout(op.tensor.axes, 3)
    elif isinstance(op, (GPUQueueSendOp, GPUQueueRecvOp)):
        return GPULayoutAssignment.generate_default_layout(op.axes, 3)
    elif isinstance(op, CTCOp):
        return GPULayoutAssignment.generate_default_layout(op.axes, 3)
    else:
        raise ValueError("Layouts not implemented for op type {}".format(op))


def gpu_constraint_factory(op, arg):
    """
    Generates a binary layout constraint given an op and an argument

    Arguments:
        op: Computation graph op which runs on the device
        arg: Argument to the op for which to generate a constraint

    Returns:
        Binary layout constraint object
    """
    if isinstance(op, AssignOp):
        return GPUEWLayoutConstraint(op, arg)
    elif isinstance(op, SetItemOp):
        return GPUSetItemLayoutConstraint(op, arg)
    elif isinstance(op, UnaryElementWiseOp):
        return GPUEWLayoutConstraint(op, arg)
    elif isinstance(op, BinaryElementWiseOp):
        return GPUEWLayoutConstraint(op, arg)
    elif isinstance(op, ReductionOp):
        return GPUEWLayoutConstraint(op, arg)
    elif isinstance(op, OneHotOp):
        return GPUEWLayoutConstraint(op, arg)
    elif isinstance(op, TensorSizeOp):
        return GPUBinaryLayoutConstraint(op, arg)
    elif isinstance(op, Fill):
        return GPUBinaryLayoutConstraint(op, arg)
    elif isinstance(op, DotOp):
        return GPUDotLayoutConstraint(op, arg)
    elif isinstance(op, ConvolutionOp):
        return GPUFixedLayoutConstraint(op, arg, arg.axes)
    elif isinstance(op, bprop_conv):
        return GPUFixedLayoutConstraint(op, arg, arg.axes)
    elif isinstance(op, update_conv):
        return GPUFixedLayoutConstraint(op, arg, arg.axes)
    elif isinstance(op, PoolingOp):
        return GPUFixedLayoutConstraint(op, arg, arg.axes)
    elif isinstance(op, BpropPoolOp):
        return GPUFixedLayoutConstraint(op, arg, arg.axes)
    elif isinstance(op, (LookupTableOp, update_lut, bprop_lut)):
        return GPULutLayoutConstraint(op, arg)
    elif isinstance(op, RngOp):
        return GPUBinaryLayoutConstraint(op, arg)
    elif isinstance(op, (GPUQueueSendOp, GPUQueueRecvOp)):
        return GPUBinaryLayoutConstraint(op, arg)
    elif isinstance(op, CTCOp):
        return GPUFixedLayoutConstraint(op, arg, arg.axes)
    else:
        raise ValueError("Layouts not implemented for op type {}".format(op))
