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
    ComputationOp, Flatten, ReorderAxes, ReductionOp, Divide
from ngraph.transformers.cpu.relu import ReluOp, BpropReluOp
from ngraph.op_graph.pooling import PoolingOp, BpropPoolOp

from ngraph.util.generics import generic_method

import ctypes as ct
import numpy as np


class MklCreateOpDescriptors(PeepholeGraphPass):
    """
    Creates MklDnn op descriptors for the ops in the graph
    Can be used by other passes to query MklDnn Engine
    Op Descriptors can also be used during primitive construction
    """

    def __init__(self, mkldnn):
        self.mkldnn = mkldnn

    @generic_method(dispatch_base_type=Op)
    def visit(self, op):
        pass

    @visit.on_type(ConvolutionOp)
    def visit(self, op):
        if self.mkldnn.mkldnn_enabled:
            input = op.args[0]
            filter = op.args[1]
            pad_d, pad_h, pad_w = itemgetter(
                *('pad_' + s for s in ('d', 'h', 'w')))(op.conv_params)
            str_d, str_h, str_w = itemgetter(
                *('str_' + s for s in ('d', 'h', 'w')))(op.conv_params)
            pad = [pad_d, pad_h, pad_w]
            stride = [str_d, str_h, str_w]
            # Only 2D convolution supported in MKLDNN for now
            if (op.args[0].axes.find_by_name('__NG_DEPTH').size != 1):
                return
            # Only single precision float supported for now
            if (op.dtype.type != np.float32):
                return
            input_shape = input.axes.lengths
            filter_shape = filter.axes.lengths
            output_shape = op.axes.lengths
            input_shape_arg = ((ct.c_int) * len(input_shape))(*input_shape)
            filter_shape_arg = ((ct.c_int) * len(filter_shape))(*filter_shape)
            output_shape_arg = ((ct.c_int) * len(output_shape))(*output_shape)
            stride_arg = ((ct.c_int) * len(stride))(*stride)
            pad_arg = ((ct.c_int) * len(pad))(*pad)
            input_layout = self.mkldnn.op_layouts.get(input.name)
            filter_layout = self.mkldnn.op_layouts.get(filter.name)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel()
            self.mkldnn.conv_fprop_kernel(
                self.mkldnn.mkldnn_engine,
                len(input_shape), len(filter_shape), len(output_shape),
                len(stride), len(pad),
                input_shape_arg, filter_shape_arg, output_shape_arg,
                stride_arg, pad_arg,
                input_layout, filter_layout,
                self.mkldnn.kernels[op.name])
            output_layout = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            if output_layout:
                self.mkldnn.op_layouts[op.name] = output_layout
            self.mkldnn.op_uses_opkernel_api[op.name] = True
            print
            print(op.name)
            self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(bprop_conv)
    def visit(self, op):
        if self.mkldnn.mkldnn_enabled:
            input = op.args[0]
            filter = op.args[1]
            pad_d, pad_h, pad_w = itemgetter(
                    *('pad_' + s for s in ('d', 'h', 'w')))(op.conv_params)
            str_d, str_h, str_w = itemgetter(
                    *('str_' + s for s in ('d', 'h', 'w')))(op.conv_params)
            pad = [pad_d, pad_h, pad_w]
            stride = [str_d, str_h, str_w]
            # Only 2D convolution supported in MKLDNN for now
            if (op.args[0].axes.find_by_name('__NG_DEPTH').size != 1):
                return
            # Only single precision float supported for now
            if (op.dtype.type != np.float32):
                return
            input_shape = input.axes.lengths
            filter_shape = filter.axes.lengths
            output_shape = op.axes.lengths
            input_shape_arg = ((ct.c_int) * len(input_shape))(*input_shape)
            filter_shape_arg = ((ct.c_int) * len(filter_shape))(*filter_shape)
            output_shape_arg = ((ct.c_int) * len(output_shape))(*output_shape)
            stride_arg = ((ct.c_int) * len(stride))(*stride)
            pad_arg = ((ct.c_int) * len(pad))(*pad)
            input_layout = self.mkldnn.op_layouts.get(input.name)
            filter_layout = self.mkldnn.op_layouts.get(filter.name)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel()
            self.mkldnn.conv_bprop_kernel(
                self.mkldnn.mkldnn_engine,
                len(input_shape), len(filter_shape), len(output_shape),
                len(stride), len(pad),
                input_shape_arg, filter_shape_arg, output_shape_arg,
                stride_arg, pad_arg,
                input_layout, filter_layout,
                self.mkldnn.kernels[op.name])
            self.mkldnn.op_layouts[op.name] = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            self.mkldnn.op_uses_opkernel_api[op.name] = True
            print
            print(op.name, " fprop:", op.fprop.forwarded.name)
            self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(update_conv)
    def visit(self, op):
        if self.mkldnn.mkldnn_enabled:
            delta = op.args[0]
            inputs = op.args[1]
            pad_d, pad_h, pad_w = itemgetter(
                    *('pad_' + s for s in ('d', 'h', 'w')))(op.conv_params)
            str_d, str_h, str_w = itemgetter(
                    *('str_' + s for s in ('d', 'h', 'w')))(op.conv_params)
            pad = [pad_d, pad_h, pad_w]
            stride = [str_d, str_h, str_w]
            # Only 2D convolution supported in MKLDNN for now
            if (delta.axes.find_by_name('__NG_DEPTH').size != 1):
                return
            # Only single precision float supported for now
            if (op.dtype.type != np.float32):
                return
            delta_shape = delta.axes.lengths
            filter_shape = op.axes.lengths
            inputs_shape = inputs.axes.lengths
            inputs_shape_arg = ((ct.c_int) * len(inputs_shape))(*inputs_shape)
            filter_shape_arg = ((ct.c_int) * len(filter_shape))(*filter_shape)
            delta_shape_arg = ((ct.c_int) * len(delta_shape))(*delta_shape)
            stride_arg = ((ct.c_int) * len(stride))(*stride)
            pad_arg = ((ct.c_int) * len(pad))(*pad)
            delta_layout = self.mkldnn.op_layouts.get(delta.name)
            filter_layout = None
            inputs_layout = self.mkldnn.op_layouts.get(inputs.name)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel()
            self.mkldnn.update_conv_kernel(
                self.mkldnn.mkldnn_engine,
                len(delta_shape), len(filter_shape), len(inputs_shape),
                len(stride), len(pad),
                delta_shape_arg, filter_shape_arg, inputs_shape_arg,
                stride_arg, pad_arg,
                delta_layout, filter_layout, inputs_layout,
                self.mkldnn.kernels[op.name])
            # self.mkldnn.op_layouts[op.name] = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            output_layout = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            if (output_layout):
                self.mkldnn.op_layouts[op.name] = output_layout
            self.mkldnn.op_uses_opkernel_api[op.name] = True
            print
            print(op.name)
            self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(ReluOp)
    def visit(self, op):
        if self.mkldnn.mkldnn_enabled:
            if (op.dtype.type != np.float32):
                return
            if (len(op.axes) != 5):
                return
            input = op.args[0]
            input_layout = self.mkldnn.op_layouts.get(input.name)
            input_size = np.prod(input.axes.lengths)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel()
            self.mkldnn.relu_fprop_kernel(
                self.mkldnn.mkldnn_engine, 
                input_size, op.slope,
                input_layout,
                self.mkldnn.kernels[op.name])
            output_layout = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            if (output_layout):
                self.mkldnn.op_layouts[op.name] = output_layout
            self.mkldnn.op_uses_opkernel_api[op.name] = True
            print
            print(op.name)
            self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(BpropReluOp)
    def visit(self, op):
        if self.mkldnn.mkldnn_enabled:
            if (op.dtype.type != np.float32):
                return
            if (len(op.axes) != 5):
                return
            delta = op.args[0]
            fprop_src = op.args[1]
            delta_layout = self.mkldnn.op_layouts.get(delta.name)
            fprop_src_layout = self.mkldnn.op_layouts.get(fprop_src.name)
            input_size = np.prod(delta.axes.lengths)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel()
            self.mkldnn.relu_bprop_kernel(
                self.mkldnn.mkldnn_engine, 
                input_size, op.fprop.forwarded.slope,
                fprop_src_layout, delta_layout,
                self.mkldnn.kernels[op.name])
            output_layout = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            if (output_layout):
                self.mkldnn.op_layouts[op.name] = output_layout
            self.mkldnn.op_uses_opkernel_api[op.name] = True
            print
            print(op.name)
            self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(PoolingOp)
    def visit(self, op):
        if (self.mkldnn.mkldnn_enabled):
            arg = op.args[0]
            input_layout = self.mkldnn.op_layouts.get(arg.name)
            C, D, H, W, N = op.axes.lengths
            input_shape = op.args[0].axes.lengths
            output_shape = op.axes.lengths
            pad_d, pad_h, pad_w = itemgetter(*('pad_' + s for s in ('d', 'h', 'w')))(op.pool_params)
            str_d, str_h, str_w = itemgetter(*('str_' + s for s in ('d', 'h', 'w')))(op.pool_params)
            pad = [pad_d, pad_h, pad_w]
            stride = [str_d, str_h, str_w]
            kernel = [op.pool_params['J'], op.pool_params['T'],
                      op.pool_params['R'], op.pool_params['S']]
            op_type = op.pool_params
            pool_type = 0
            if op_type['op'] == 'avg':
                pool_type = 1
            [J, T, R, S] = kernel
            # Only 2D pooling supported in MKLDNN for now
            if (D != 1 or T != 1 or J != 1):
                return
            # Only single precision float supported for now
            if op.dtype != np.float32:
                return
            # Sanity check tensor shapes
            if ((len(op.axes.lengths) != 5) or
                    (len(stride) != 3) or (len(pad) != 3)):
                return
            input_shape_arg = ((ct.c_int) * len(input_shape))(*input_shape)
            output_shape_arg = ((ct.c_int) * len(output_shape))(*output_shape)
            kernel_sizes = ((ct.c_int) * len(kernel))(*kernel)
            pad_data = ((ct.c_int) * len(pad))(*pad)
            stride_data = ((ct.c_int) * len(stride))(*stride)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel()
            self.mkldnn.pool_fprop_kernel(
                self.mkldnn.mkldnn_engine,
                len(input_shape), len(output_shape), len(stride), len(pad),
                input_shape_arg, kernel_sizes, output_shape_arg,
                stride_data, pad_data, pool_type,
                input_layout, self.mkldnn.kernels[op.name])
            output_layout = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            if (output_layout):
                self.mkldnn.op_layouts[op.name] = output_layout
            self.mkldnn.op_uses_opkernel_api[op.name] = True
            print
            print(op.name)
            self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(BpropPoolOp)
    def visit(self, op):
        if (self.mkldnn.mkldnn_enabled):
            arg = op.args[0]
            input_layout = self.mkldnn.op_layouts.get(arg.name)
            C, D, H, W, N = op.axes.lengths
            input_shape = op.args[0].axes.lengths
            output_shape = op.axes.lengths
            pad_d, pad_h, pad_w = itemgetter(*('pad_' + s for s in ('d', 'h', 'w')))(op.pool_params)
            str_d, str_h, str_w = itemgetter(*('str_' + s for s in ('d', 'h', 'w')))(op.pool_params)
            pad = [pad_d, pad_h, pad_w]
            stride = [str_d, str_h, str_w]
            kernel = [op.pool_params['J'], op.pool_params['T'],
                      op.pool_params['R'], op.pool_params['S']]
            op_type = op.pool_params
            pool_type = 0
            if op_type['op'] == 'avg':
                pool_type = 1
            [J, T, R, S] = kernel
            # Only 2D pooling supported in MKLDNN for now
            if (D != 1 or T != 1 or J != 1):
                return
            # Only single precision float supported for now
            if op.dtype != np.float32:
                return
            # Sanity check tensor shapes
            if ((len(op.axes.lengths) != 5) or
                    (len(stride) != 3) or (len(pad) != 3)):
                return
            input_shape_arg = ((ct.c_int) * len(input_shape))(*input_shape)
            output_shape_arg = ((ct.c_int) * len(output_shape))(*output_shape)
            kernel_sizes = ((ct.c_int) * len(kernel))(*kernel)
            pad_data = ((ct.c_int) * len(pad))(*pad)
            stride_data = ((ct.c_int) * len(stride))(*stride)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel()
            self.mkldnn.pool_bprop_kernel(
                self.mkldnn.mkldnn_engine,
                len(input_shape), len(output_shape), len(stride), len(pad),
                input_shape_arg, kernel_sizes, output_shape_arg,
                stride_data, pad_data, pool_type,
                input_layout, self.mkldnn.kernels[op.fprop.forwarded.name],
                self.mkldnn.kernels[op.name])
            output_layout = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            if (output_layout):
                self.mkldnn.op_layouts[op.name] = output_layout
            self.mkldnn.op_uses_opkernel_api[op.name] = True
            print
            print(op.name)
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

    def __init__(self, mkldnn):
        self.mkldnn = mkldnn
        self.reorder_ops = dict()   # Maps op.name to reorder op

    def init_mkldnn_reorder(self, op):
        assert len(op.axes) ==5
        output_shape = op.axes.lengths
        (C, D, H, W, N) = op.axes.lengths
        assert D == 1
        output_sizes = (N, C, H, W)
        output_sizes_arg = ((ct.c_int) * len(output_sizes))(*output_sizes)
        self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel()
        self.mkldnn.reorder_kernel(
            self.mkldnn.mkldnn_engine,
            len(output_sizes), output_sizes_arg,
            1, # mkldnn_f32 in mkldnn_types.h. TODO(jbobba): find a better way
            7, # mkldnn_chwn in mkldnn_types.h
            op.in_layout, None,
            self.mkldnn.kernels[op.name]
        )
        self.mkldnn.op_uses_opkernel_api[op.name] = True
        print
        print(op.name)
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
        for arg in op.args:
            if arg.name in self.mkldnn.op_layouts:
                pass

    @visit.on_type(Divide)
    def visit(self, op):
        for arg in op.args:
            if arg.name in self.mkldnn.op_layouts:
                pass

    @visit.on_type(ReorderAxes)
    def visit(self, op):
        arg = op.args[0]
        if arg.name in self.mkldnn.op_layouts:
            reorder_op = self.get_reorder_op(arg)
            self.replace_op(op, ReorderAxes(reorder_op, axes=op.axes))

    @visit.on_type(ReductionOp)
    def visit(self, op):
        arg = op.args[0]
        if arg.name in self.mkldnn.op_layouts:
            reorder_op = self.get_reorder_op(arg)
            self.replace_op(op, ReductionOp(reorder_op, axes=op.axes))

    @visit.on_type(ReluOp)
    def visit(self, op):
        if op.name in self.mkldnn.kernels:
            pass
        else:
            arg = op.args[0]
            if arg.name in self.mkldnn.op_layouts:
                reorder_op = MklReorderOp(arg, in_layout=self.mkldnn.op_layouts[arg.name], out_layout=None)
                self.init_mkldnn_reorder(reorder_op)
                self.replace_op(op, ReluOp(reorder_op, op.slope))

    @visit.on_type(BpropReluOp)
    def visit(self, op):
        if op.name in self.mkldnn.kernels:
            pass
        else:
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
                self.replace_op(op, BpropReluOp(new_args[0], new_args[1], op.fprop))

    @visit.on_type(ConvolutionOp)
    def visit(self, op):
        if op.name in self.mkldnn.kernels:
            pass
        else:
            arg = op.args[0]
            if arg.name in self.mkldnn.op_layouts:
                assert(0)

    @visit.on_type(bprop_conv)
    def visit(self, op):
        if op.name in self.mkldnn.kernels:
            pass
        else:
            arg = op.args[0]
            if arg.name in self.mkldnn.op_layouts:
                assert (0)

    @visit.on_type(BroadcastOp)
    def visit(self, op):
        arg = op.args[0]
        if arg.name in self.mkldnn.op_layouts:
            reorder_op = self.get_reorder_op(arg)
            self.replace_op(op, BroadcastOp(reorder_op, axes=op.axes))

    @visit.on_type(MapRolesOp)
    def visit(self, op):
        pass

    @visit.on_type(MklReorderOp)
    def visit(self, op):
        pass

    @visit.on_type(update_conv)
    def visit(self, op):
        if op.name in self.mkldnn.kernels:
            pass
        else:
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
                filters = op.fprop.args[1]
                self.replace_op(op, update_conv(new_args[0], new_args[1], filters, op.fprop))

    @visit.on_type(PoolingOp)
    def visit(self, op):
        if op.name in self.mkldnn.kernels:
            pass
        else:
            arg = op.args[0]
            if arg.name in self.mkldnn.op_layouts:
                reorder_op = self.get_reorder_op(arg)
                self.replace_op(op, PoolingOp(op.pool_params, reorder_op, axes=op.axes))

    @visit.on_type(BpropPoolOp)
    def visit(self, op):
        if op.name in self.mkldnn.kernels:
            pass
        else:
            arg = op.args[0]
            if arg.name in self.mkldnn.op_layouts:
                reorder_op = self.get_reorder_op(arg)
                self.replace_op(op, BpropPoolOp(reorder_op, op.fprop.args[0], op.fprop))

    @visit.on_type(Flatten)
    def visit(self, op):
        arg = op.args[0]
        if arg.name in self.mkldnn.op_layouts:
            reorder_op = self.get_reorder_op(arg)
            self.replace_op(op, Flatten(reorder_op, op.axes))

    @visit.on_type(ComputationOp)
    def visit(self, op):
        return
        if isinstance(self.computation.returns, Op):
            return value(self.computation.returns)
        elif isinstance(self.computation.returns, (collections.Sequence, OrderedSet)):
            return tuple(value(op) for op in self.computation.returns)
        elif isinstance(self.computation.returns, collections.Set):
            result = dict()
            for op in self.computation.returns:
                result[op] = value(op)
            return result
        else:
            return None

