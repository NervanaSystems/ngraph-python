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
from operator import itemgetter

from ngraph.transformers.passes.passes import PeepholeGraphPass
from ngraph.op_graph.convolution import ConvolutionOp, bprop_conv, update_conv
from ngraph.op_graph.op_graph import Op, MapRolesOp, TensorOp, BroadcastOp, \
    ComputationOp, Flatten, unflatten, ReorderAxes, ReductionOp, Divide
from ngraph.transformers.cpu.relu import ReluOp, BpropReluOp
from ngraph.op_graph.pooling import PoolingOp, BpropPoolOp
from ngraph.op_graph.batchnorm import BatchnormOp
from ngraph.transformers.passes.layout import AddLayoutConversions


from ngraph.util.generics import generic_method

import ctypes as ct
import numpy as np
import collections
from orderedset import OrderedSet


class MklCreateOpDescriptors(PeepholeGraphPass):
    """
    Creates MklDnn op descriptors for the ops in the graph
    Can be used by other passes to query MklDnn Engine
    Op Descriptors can also be used during primitive construction
    """

    def __init__(self, mkldnn):
        self.mkldnn = mkldnn

    def get_data_shape(self, axes):
        [C, D, H, W, N] = axes.lengths
        return [N, C, H, W]
        #N = axes.batch_axis().length
        #C = axes.channel_axis().length
        #spatial_sizes = [axis.length for axis in axes.spatial_axes()]
        #if len(spatial_sizes) == 3:
        #    return [N, C, spatial_sizes[1], spatial_sizes[2]]
        #else:
        #    return [N, C, spatial_sizes[0], spatial_sizes[1]]

    def get_filter_shape(self, axes):
        [I, D, H, W, O] = axes.lengths
        return [O, I, H, W]

    @generic_method(dispatch_base_type=Op)
    def visit(self, op):
        pass

    @visit.on_type(BatchnormOp)
    def visit(self, op):
        if self.mkldnn.mkldnn_enabled:
            inputs = op.args[0]
            gamma = op.args[1]
            bias = op.args[2]
            mean = op.args[4]
            variance = op.args[5]
            # unflatten the inputs and extract C H W N params
            unflatten_inputs = unflatten(inputs)
            # Only single precision float supported for now
            if op.dtype != np.float32:
                return
            # Sanity check tensor shapes
            if (len(unflatten_inputs.axes.lengths) != 5):
                return
            data_type = self.mkldnn.datatype[op.dtype.type]
            C, D, H, W, N = unflatten_inputs.axes.lengths
            inputs_shape = (C, H, W, N)
            mean_size = mean.axes.lengths[0]
            gamma_shape = gamma.axes.lengths[0]
            bias_shape = bias.axes.lengths[0]
            variance_size = variance.axes.lengths[0]
            outputs_shape = op.axes.lengths

            # weights is 2 dimensional, 1-st dimension contains gamma parameter, 2-nd dimension contains beta parameter.
            weights_shape = (2, gamma_shape)
            weights_shape_arg = ((ct.c_int) * len(weights_shape))(*weights_shape)

            input_shape_arg = ((ct.c_int) * len(inputs_shape))(*inputs_shape)
            outputs_shape_arg = ((ct.c_int) * len(outputs_shape))(*outputs_shape)

            inputs_layout = None
            if inputs.name in self.mkldnn.kernels:
                input_layout = self.mkldnn.op_layouts[inputs.name]

            mean_layout = None
            if mean.name in self.mkldnn.kernels:
                mean_layout = self.mkldnn.op_layouts[mean.name]

            variance_layout = None
            if variance.name in self.mkldnn.kernels:
                variance_layout = self.mkldnn.op_layouts[variance.name]
            op_id = len(self.mkldnn.kernels)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel(op_id)

            self.mkldnn.batchnorm_fprop_kernel(
                self.mkldnn.mkldnn_engine, len(inputs_shape), len(weights_shape),
                len(outputs_shape), mean_size, variance_size, input_shape_arg, weights_shape_arg, outputs_shape_arg,
                op.eps, inputs_layout, None, mean_layout, variance_layout, data_type, self.mkldnn.kernels[op.name])
            output_layout = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            if output_layout:
                self.mkldnn.op_layouts[op.name] = output_layout
            self.mkldnn.op_uses_opkernel_api[op.name] = True
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(ConvolutionOp)
    def visit(self, op):
        if self.mkldnn.mkldnn_enabled:
            input = op.args[0]
            filter = op.args[1]
            # Only 2D convolution supported in MKLDNN for now
            if (op.args[0].axes.find_by_name('__NG_DEPTH').size != 1):
                return
            # Only single precision float supported for now
            if (op.dtype.type != np.float32):
                return

            data_type = self.mkldnn.datatype[op.dtype.type]
            input_shape = self.get_data_shape(input.axes)
            filter_shape = self.get_filter_shape(filter.axes)
            output_shape = self.get_data_shape(op.axes)
            pad_d, pad_h, pad_w = itemgetter(
                *('pad_' + s for s in ('d', 'h', 'w')))(op.conv_params)
            str_d, str_h, str_w = itemgetter(
                *('str_' + s for s in ('d', 'h', 'w')))(op.conv_params)
            pad = [pad_h, pad_w]
            stride = [str_h, str_w]
            print ("Input: ", input_shape, " Filter: ", filter_shape, " Output: ", output_shape, " Pad: ", pad, " Stride: ", stride)
            input_shape_arg = ((ct.c_int) * len(input_shape))(*input_shape)
            filter_shape_arg = ((ct.c_int) * len(filter_shape))(*filter_shape)
            output_shape_arg = ((ct.c_int) * len(output_shape))(*output_shape)
            stride_arg = ((ct.c_int) * len(stride))(*stride)
            pad_arg = ((ct.c_int) * len(pad))(*pad)
            input_layout = self.mkldnn.op_layouts.get(input.name)
            filter_layout = self.mkldnn.op_layouts.get(filter.name)

            op_id = len(self.mkldnn.kernels)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel(op_id)
            self.mkldnn.conv_fprop_kernel(
                self.mkldnn.mkldnn_engine,
                len(input_shape), len(filter_shape), len(output_shape),
                input_shape_arg, filter_shape_arg, output_shape_arg,
                stride_arg, pad_arg,
                input_layout, filter_layout,
                data_type,
                self.mkldnn.kernels[op.name])
            output_layout = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            if output_layout:
                self.mkldnn.op_layouts[op.name] = output_layout
            self.mkldnn.op_uses_opkernel_api[op.name] = True
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(bprop_conv)
    def visit(self, op):
        if self.mkldnn.mkldnn_enabled:
            input = op.args[0]
            filter = op.args[1]
            # Only 2D convolution supported in MKLDNN for now
            if (op.args[0].axes.find_by_name('__NG_DEPTH').size != 1):
                return
            # Only single precision float supported for now
            if (op.dtype.type != np.float32):
                return

            data_type = self.mkldnn.datatype[op.dtype.type]
            input_shape = self.get_data_shape(input.axes)
            filter_shape = self.get_filter_shape(filter.axes)
            output_shape = self.get_data_shape(op.axes)
            pad_d, pad_h, pad_w = itemgetter(
                *('pad_' + s for s in ('d', 'h', 'w')))(op.conv_params)
            str_d, str_h, str_w = itemgetter(
                *('str_' + s for s in ('d', 'h', 'w')))(op.conv_params)
            pad = [pad_h, pad_w]
            stride = [str_h, str_w]

            input_shape_arg = ((ct.c_int) * len(input_shape))(*input_shape)
            filter_shape_arg = ((ct.c_int) * len(filter_shape))(*filter_shape)
            output_shape_arg = ((ct.c_int) * len(output_shape))(*output_shape)
            stride_arg = ((ct.c_int) * len(stride))(*stride)
            pad_arg = ((ct.c_int) * len(pad))(*pad)
            input_layout = self.mkldnn.op_layouts.get(input.name)
            filter_layout = self.mkldnn.op_layouts.get(filter.name)
            op_id = len(self.mkldnn.kernels)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel(op_id)
            self.mkldnn.conv_bprop_kernel(
                self.mkldnn.mkldnn_engine,
                len(input_shape), len(filter_shape), len(output_shape),
                input_shape_arg, filter_shape_arg, output_shape_arg,
                stride_arg, pad_arg,
                input_layout, filter_layout,
                data_type,
                self.mkldnn.kernels[op.name])
            self.mkldnn.op_layouts[op.name] = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            self.mkldnn.op_uses_opkernel_api[op.name] = True
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(update_conv)
    def visit(self, op):
        if self.mkldnn.mkldnn_enabled:
            delta = op.args[0]
            inputs = op.args[1]
            # Only 2D convolution supported in MKLDNN for now
            if (delta.axes.find_by_name('__NG_DEPTH').size != 1):
                return
            # Only single precision float supported for now
            if (op.dtype.type != np.float32):
                return

            data_type = self.mkldnn.datatype[op.dtype.type]
            delta_shape = self.get_data_shape(delta.axes)
            filter_shape = self.get_filter_shape(op.axes)
            inputs_shape = self.get_data_shape(inputs.axes)
            pad_d, pad_h, pad_w = itemgetter(
                *('pad_' + s for s in ('d', 'h', 'w')))(op.conv_params)
            str_d, str_h, str_w = itemgetter(
                *('str_' + s for s in ('d', 'h', 'w')))(op.conv_params)
            pad = [pad_h, pad_w]
            stride = [str_h, str_w]

            inputs_shape_arg = ((ct.c_int) * len(inputs_shape))(*inputs_shape)
            filter_shape_arg = ((ct.c_int) * len(filter_shape))(*filter_shape)
            delta_shape_arg = ((ct.c_int) * len(delta_shape))(*delta_shape)
            stride_arg = ((ct.c_int) * len(stride))(*stride)
            pad_arg = ((ct.c_int) * len(pad))(*pad)
            delta_layout = self.mkldnn.op_layouts.get(delta.name)
            filter_layout = None
            inputs_layout = self.mkldnn.op_layouts.get(inputs.name)
            op_id = len(self.mkldnn.kernels)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel(op_id)
            self.mkldnn.update_conv_kernel(
                self.mkldnn.mkldnn_engine,
                len(delta_shape), len(filter_shape), len(inputs_shape),
                delta_shape_arg, filter_shape_arg, inputs_shape_arg,
                stride_arg, pad_arg,
                delta_layout, filter_layout, inputs_layout,
                data_type,
                self.mkldnn.kernels[op.name])
            output_layout = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            if (output_layout):
                self.mkldnn.op_layouts[op.name] = output_layout
            self.mkldnn.op_uses_opkernel_api[op.name] = True
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(ReluOp)
    def visit(self, op):
        if self.mkldnn.mkldnn_enabled:
            if (op.dtype.type != np.float32):
                return
            if (len(op.axes) != 5):
                return
            data_type = self.mkldnn.datatype[op.dtype.type]
            input = op.args[0]
            input_layout = self.mkldnn.op_layouts.get(input.name)
            input_size = np.prod(input.axes.lengths)
            op_id = len(self.mkldnn.kernels)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel(op_id)
            self.mkldnn.relu_fprop_kernel(
                self.mkldnn.mkldnn_engine, 
                input_size, op.slope,
                input_layout,
                data_type,
                self.mkldnn.kernels[op.name])
            output_layout = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            if (output_layout):
                self.mkldnn.op_layouts[op.name] = output_layout
            self.mkldnn.op_uses_opkernel_api[op.name] = True
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(BpropReluOp)
    def visit(self, op):
        if self.mkldnn.mkldnn_enabled:
            if (op.dtype.type != np.float32):
                return
            if (len(op.axes) != 5):
                return
            data_type = self.mkldnn.datatype[op.dtype.type]
            delta = op.args[0]
            fprop_src = op.args[1]
            delta_layout = self.mkldnn.op_layouts.get(delta.name)
            fprop_src_layout = self.mkldnn.op_layouts.get(fprop_src.name)
            input_size = np.prod(delta.axes.lengths)
            op_id = len(self.mkldnn.kernels)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel(op_id)
            self.mkldnn.relu_bprop_kernel(
                self.mkldnn.mkldnn_engine, 
                input_size, op.fprop.forwarded.slope,
                fprop_src_layout, delta_layout,
                data_type,
                self.mkldnn.kernels[op.name])
            output_layout = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            if (output_layout):
                self.mkldnn.op_layouts[op.name] = output_layout
            self.mkldnn.op_uses_opkernel_api[op.name] = True
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(PoolingOp)
    def visit(self, op):
        if (self.mkldnn.mkldnn_enabled):
            input = op.args[0]
            # Only 2D pooling supported in MKLDNN for now
            if (input.axes.find_by_name('__NG_DEPTH').size != 1):
                return
            if (op.pool_params['J'] != 1 or op.pool_params['T'] != 1):
                return
            # Only single precision float supported for now
            if op.dtype != np.float32:
                return
            # Sanity check tensor shapes
            if (len(op.axes.lengths) != 5):
                return
            
            data_type = self.mkldnn.datatype[op.dtype.type]
            input_shape = self.get_data_shape(input.axes)
            output_shape = self.get_data_shape(op.axes)
            kernel = [op.pool_params['R'], op.pool_params['S']]
            pad_d, pad_h, pad_w = itemgetter(*('pad_' + s for s in ('d', 'h', 'w')))(op.pool_params)
            str_d, str_h, str_w = itemgetter(*('str_' + s for s in ('d', 'h', 'w')))(op.pool_params)
            pad = [pad_h, pad_w]
            stride = [str_h, str_w]
            op_type = op.pool_params
            pool_type = 0
            if op_type['op'] == 'avg':
                pool_type = 1
            input_shape_arg = ((ct.c_int) * len(input_shape))(*input_shape)
            output_shape_arg = ((ct.c_int) * len(output_shape))(*output_shape)
            kernel_sizes = ((ct.c_int) * len(kernel))(*kernel)
            pad_data = ((ct.c_int) * len(pad))(*pad)
            stride_data = ((ct.c_int) * len(stride))(*stride)
            input_layout = self.mkldnn.op_layouts.get(input.name)
            
            op_id = len(self.mkldnn.kernels)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel(op_id)
            self.mkldnn.pool_fprop_kernel(
                self.mkldnn.mkldnn_engine,
                len(input_shape), len(output_shape),
                input_shape_arg, kernel_sizes, output_shape_arg,
                stride_data, pad_data, pool_type,
                input_layout, data_type, self.mkldnn.kernels[op.name])
            output_layout = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            if (output_layout):
                self.mkldnn.op_layouts[op.name] = output_layout
            self.mkldnn.op_uses_opkernel_api[op.name] = True
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(BpropPoolOp)
    def visit(self, op):
        if (self.mkldnn.mkldnn_enabled):
            input = op.args[0]
            # Only 2D pooling supported in MKLDNN for now
            if (input.axes.find_by_name('__NG_DEPTH').size != 1):
                return
            if (op.pool_params['J'] != 1 or op.pool_params['T'] != 1):
                return
            # Only single precision float supported for now
            if op.dtype != np.float32:
                return
            # Sanity check tensor shapes
            if (len(op.axes.lengths) != 5):
                return
            
            data_type = self.mkldnn.datatype[op.dtype.type]
            input_shape = self.get_data_shape(input.axes)
            output_shape = self.get_data_shape(op.axes)
            kernel = [op.pool_params['R'], op.pool_params['S']]
            pad_d, pad_h, pad_w = itemgetter(*('pad_' + s for s in ('d', 'h', 'w')))(op.pool_params)
            str_d, str_h, str_w = itemgetter(*('str_' + s for s in ('d', 'h', 'w')))(op.pool_params)
            pad = [pad_h, pad_w]
            stride = [str_h, str_w]
            op_type = op.pool_params
            pool_type = 0
            if op_type['op'] == 'avg':
                pool_type = 1
            input_shape_arg = ((ct.c_int) * len(input_shape))(*input_shape)
            output_shape_arg = ((ct.c_int) * len(output_shape))(*output_shape)
            kernel_sizes = ((ct.c_int) * len(kernel))(*kernel)
            pad_data = ((ct.c_int) * len(pad))(*pad)
            stride_data = ((ct.c_int) * len(stride))(*stride)
            input_layout = self.mkldnn.op_layouts.get(input.name)
            
            op_id = len(self.mkldnn.kernels)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel(op_id)
            self.mkldnn.pool_bprop_kernel(
                self.mkldnn.mkldnn_engine,
                len(input_shape), len(output_shape),
                input_shape_arg, kernel_sizes, output_shape_arg,
                stride_data, pad_data, pool_type,
                input_layout, data_type,
                self.mkldnn.kernels[op.fprop.forwarded.name],
                self.mkldnn.kernels[op.name])
            output_layout = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            if (output_layout):
                self.mkldnn.op_layouts[op.name] = output_layout
            self.mkldnn.op_uses_opkernel_api[op.name] = True
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(MapRolesOp)
    def visit(self, op):
        arg = op.args[0]
        if arg.name in self.mkldnn.op_layouts:
            self.mkldnn.op_layouts[op.name] = self.mkldnn.op_layouts[arg.name]

class MklReorderOp(TensorOp):

    def __init__(self, arg, in_layout, out_layout, **kwargs):
        super(MklReorderOp, self).__init__(args=(arg,), axes=arg.axes, **kwargs)
        self.in_layout = in_layout
        self.out_layout = out_layout


class MklAddLayoutConversions(PeepholeGraphPass):
    """
    Adds layout conversion nodes when an MKLDNN tensor is utilized by a 
    non-MKL op 
    """

    def __init__(self, mkldnn, layoutpass):
        self.mkldnn = mkldnn
        self.layoutpass = layoutpass
        self.reorder_ops = dict()   # Maps op.name to reorder op

    def init_mkldnn_reorder(self, op):
        assert len(op.axes) ==5
        output_shape = op.axes.lengths
        (C, D, H, W, N) = op.axes.lengths
        assert D == 1
        output_sizes = (N, C, H, W)
        output_sizes_arg = ((ct.c_int) * len(output_sizes))(*output_sizes)
        op_id = len(self.mkldnn.kernels)
        self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel(op_id)
        self.mkldnn.reorder_kernel(
            self.mkldnn.mkldnn_engine,
            len(output_sizes), output_sizes_arg,
            self.mkldnn.datatype[op.dtype.type],
            self.mkldnn.memory_format['chwn'],
            op.in_layout, None,
            self.mkldnn.kernels[op.name]
        )
        self.mkldnn.op_uses_opkernel_api[op.name] = True
        if self.mkldnn.mkldnn_verbose:
            print
            print(op_id, op.name)
            self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    def get_reorder_op(self, op):
        if op.name in self.reorder_ops:
            return self.reorder_ops[op.name]
        else:
            reorder_op = MklReorderOp(op, in_layout=self.mkldnn.op_layouts[op.name], out_layout=None)
            self.reorder_ops[op.name] = reorder_op
            self.init_mkldnn_reorder(reorder_op)
            return reorder_op

    @generic_method(dispatch_base_type=Op)
    def visit(self, op):
        if op.name in self.mkldnn.kernels:
            return
        replace = False
        new_args = []
        for arg in op.args:
            if arg.name in self.mkldnn.op_layouts:
                reorder_op = self.get_reorder_op(arg)
                new_args.append(reorder_op)
                replace = True
            else:
                new_args.append(arg)
        if replace:
            new_op = self.layoutpass.op_from_args(op, new_args)
            self.replace_op(op, new_op)

    @visit.on_type(MapRolesOp)
    def visit(self, op):
        pass

    @visit.on_type(MklReorderOp)
    def visit(self, op):
        pass

    @visit.on_type(ComputationOp)
    def visit(self, op):
        if isinstance(op.returns, Op) and op.returns.name in self.mkldnn.op_layouts:
            reorder_op = self.get_reorder_op(op.returns.forwarded)
            op.returns = reorder_op
            op.add_control_dep(reorder_op)
        elif isinstance(op.returns, (collections.Sequence, OrderedSet)):
            returns = op.returns
            op.returns = []
            for orig_op in returns:
                if orig_op.forwarded.name in self.mkldnn.op_layouts:
                    reorder_op = self.get_reorder_op(orig_op.forwarded)
                    op.returns.append(reorder_op)
                    op.add_control_dep(reorder_op)
                else:
                    op.returns.append(orig_op)
        elif isinstance(op.returns, collections.Set):
            # TODO(jbobba): Handle this case
            assert False
            returns = op.returns
            op.returns = []
            for orig_op in returns:
                if orig_op.forwarded.name in self.mkldnn.op_layouts:
                    reorder_op = self.get_reorder_op(orig_op.forwarded)
                    op.returns.append(reorder_op)
                    op.add_control_dep(reorder_op)
                else:
                    op.returns.append(orig_op)
        else:
            pass

