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
from ngraph.op_graph.op_graph import Op, MapRolesOp, TensorOp, BroadcastOp, ComputationOp, Flatten
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
            input_layout = None
            if input.name in self.mkldnn.kernels:
                input_layout = self.mkldnn.op_layouts[input.name]
            filter_layout = None
            if filter.name in self.mkldnn.kernels:
                filter_layout = self.mkldnn.op_layouts[filter.name]
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel()
            self.mkldnn.conv_fprop_kernel(
                self.mkldnn.mkldnn_engine,
                len(input_shape), len(filter_shape), len(output_shape),
                len(stride), len(pad),
                input_shape_arg, filter_shape_arg, output_shape_arg,
                stride_arg, pad_arg,
                input_layout, filter_layout,
                self.mkldnn.kernels[op.name])
            self.mkldnn.op_layouts[op.name] = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
            self.mkldnn.op_uses_opkernel_api[op.name] = True

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
            input_layout = None
            if input.name in self.mkldnn.kernels:
                input_layout = self.mkldnn.op_layouts[input.name]
            filter_layout = None
            if filter.name in self.mkldnn.kernels:
                filter_layout = self.mkldnn.op_layouts[filter.name]
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

    @visit.on_type(ReluOp)
    def visit(self, op):
        if self.mkldnn.mkldnn_enabled:
            if (op.dtype.type != np.float32):
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

    @visit.on_type(BpropReluOp)
    def visit(self, op):
        if self.mkldnn.mkldnn_enabled:
            if (op.dtype.type != np.float32):
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

    @visit.on_type(PoolingOp)
    def visit(self, op):
        if (self.mkldnn.mkldnn_enabled):
            arg = op.args[0]
            input_layout = None
            if arg.name in self.mkldnn.op_layouts:
                input_layout = self.mkldnn.op_layouts[arg.name]
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

    @visit.on_type(BpropPoolOp)
    def visit(self, op):
        if (self.mkldnn.mkldnn_enabled):
            arg = op.args[0]
            input_layout = None
            if arg.name in self.mkldnn.op_layouts:
                input_layout = self.mkldnn.op_layouts[arg.name]
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

    def init_mkldnn_reorder(self, op):
        output_shape = op.axes.lengths
        assert op.axes.find_by_name('__NG_DEPTH').size == 1
        output_shape_arg = ((ct.c_int) * len(output_shape))(*output_shape)
        self.mkldnn.kernels[op.name] = self.mkldnn.create_mkldnn_netlist_fn()
        self.mkldnn.create_reorder_kernel_fn(
            self.mkldnn.mkldnn_engine,
            op.in_layout,
            len(output_shape),
            output_shape_arg,
            self.mkldnn.kernels[op.name]
        )

    @generic_method(dispatch_base_type=Op)
    def visit(self, op):
        for arg in op.args:
            if arg.name in self.mkldnn.op_layouts:
                pass
                #reorder_op = MklReorderOp(arg, in_layout=self.mkldnn.op_layouts[arg.name], out_layout=None)
                #self.init_mkldnn_reorder(reorder_op)

    @visit.on_type(BroadcastOp)
    def visit(self, op):
        arg = op.args[0]
        if arg.name in self.mkldnn.op_layouts:
            reorder_op = MklReorderOp(arg, in_layout=self.mkldnn.op_layouts[arg.name], out_layout=None)
            self.init_mkldnn_reorder(reorder_op)
            self.replace_op(op, BroadcastOp(reorder_op, axes=op.axes))

    @visit.on_type(MapRolesOp)
    def visit(self, op):
        pass

    @visit.on_type(MklReorderOp)
    def visit(self, op):
        pass

    @visit.on_type(update_conv)
    def visit(self, op):
        replace = False
        new_args = []
        for arg in op.args:
            if arg.name in self.mkldnn.op_layouts:
                reorder_op = MklReorderOp(arg, in_layout=self.mkldnn.op_layouts[arg.name], out_layout=None)
                self.init_mkldnn_reorder(reorder_op)
                new_args.append(reorder_op)
                replace = True
            else:
                new_args.append(arg)
        if replace:
            filters = op.fprop.args[1]
            self.replace_op(op, update_conv(new_args[0], new_args[1], filters, op.fprop))

    @visit.on_type(PoolingOp)
    def visit(self, op):
        arg = op.args[0]
        #if arg.name in self.mkldnn.op_layouts:
            #self.mkldnn.op_layouts[op.name] = self.mkldnn.op_layouts[arg.name]
            #reorder_op = MklReorderOp(arg, in_layout=self.mkldnn.op_layouts[arg.name], out_layout=None)
            #self.init_mkldnn_reorder(reorder_op)
            #self.replace_op(op, PoolingOp(op.pool_params, reorder_op, axes=op.axes))

    @visit.on_type(BpropPoolOp)
    def visit(self, op):
        arg = op.args[0]

    @visit.on_type(Flatten)
    def visit(self, op):
        arg = op.args[0]
        if arg.name in self.mkldnn.op_layouts:
            reorder_op = MklReorderOp(arg, in_layout=self.mkldnn.op_layouts[arg.name], out_layout=None)
            self.init_mkldnn_reorder(reorder_op)
            self.replace_op(op, Flatten(reorder_op, op.axes))
