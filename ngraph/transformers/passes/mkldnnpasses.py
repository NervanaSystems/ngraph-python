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
    ComputationOp, Flatten, unflatten, ReorderAxes, ReductionOp, Divide, \
    DotLowDimension, Add, ContiguousOp
from ngraph.transformers.cpu.relu import ReluOp, BpropReluOp
from ngraph.op_graph.pooling import PoolingOp, BpropPoolOp
from ngraph.op_graph.batchnorm import BatchnormOp
from ngraph.op_graph.axes import Axes
from ngraph.transformers.passes.layout import AddLayoutConversions


from ngraph.util.generics import generic_method

import ctypes as ct
import numpy as np
import collections
from orderedset import OrderedSet


class MklReorderOp(TensorOp):
    '''
    Converts op value tensor from MKL layouts to "native" layout
    '''
    def __init__(self, arg, in_layout, out_layout, **kwargs):
        super(MklReorderOp, self).__init__(args=(arg,), axes=arg.axes, **kwargs)
        self.in_layout = in_layout
        self.out_layout = out_layout

def get_order_from_axes(axes, sub_axes):
    order = []
    for a in sub_axes:
      found = False
      for (index, b) in enumerate(axes):
        if b.name == a.name:
          order.append(index)
          found = True
          continue
      if not found:
        assert False, "Axis not found"
    return order

def get_axes_mkl_order(axes, order):
    return [axes[index] for index in order]

def get_size_mkl_order(axes, order):
    return [a.length for a in get_axes_mkl_order(axes, order)]

def get_strides_mkl_order(td, order):
  return [td.strides[index] for index in order]

def get_native_layout(mkldnn, td, order, use_formats=False):
  '''
  Create an MKL layout object in transformer-visible layout 
  :param td: tensor description of the op. Currently owns tensor layout info in graph
  :param order: order in which axes need to be specified to MKL
  :param use_formats: Optimization to identify canned MKL formats
  :return: MKL layout object
  '''
  op_axes = td.axes
  mkl_shape = get_size_mkl_order(op_axes, order)
  data_type = mkldnn.datatype[td.dtype.type]
  elem_size = td.dtype.itemsize
  mkl_strides = [stride/elem_size for stride in get_strides_mkl_order(td, order)]
  # TODO(jbobba) - Handle views for tensors that are not fully materialized
  mkl_axes = [axis for axis in get_axes_mkl_order(op_axes, order)]
  mkl_shape_arg = ((ct.c_int) * len(mkl_shape))(*mkl_shape)
  mkl_strides_arg = ((ct.c_int) * len(mkl_strides))(*mkl_strides)
  memory_format = mkldnn.memory_format['blocked']
  if use_formats:
    # Look for canned formats
    if len(mkl_strides) == 4:
      [N, C, H, W] = mkl_strides
      stride_order = sorted([N, C, H, W], reverse=True)
      if (stride_order == [C, H, W, N]):
        memory_format = mkldnn.memory_format['chwn']
      elif (stride_order == [N, C, H, W]):
        memory_format = mkldnn.memory_format['nchw']
    elif len(mkl_strides) == 2:
      [N, C] = mkl_strides
      stride_order = sorted([N, C], reverse=True)
      if stride_order == [N, C]:
        memory_format = mkldnn.memory_format['nc']

  native_layout = mkldnn.create_layout_pd(
    mkldnn.mkldnn_engine,
    len(mkl_shape), mkl_shape_arg,
    mkl_strides_arg, data_type, memory_format)
  mkldnn.native_layouts[td] = native_layout
  return (native_layout, mkl_axes)

def get_mkl_layout(mkldnn, op, order, use_formats=False):
    if op.name in mkldnn.op_layouts:
        return mkldnn.op_layouts[op.name]
    else: 
        return get_native_layout(mkldnn, op.tensor_description(), order, use_formats)

class MklCreateOpDescriptors(PeepholeGraphPass):
    """
    Creates MKL-DNN op kernels for ops in the graph that have an MKL-DNN implementation.
    Most MKL-DNN op kernels produce tensors in MKL-DNN layout that is tracked and propagated 
    to downstream ops. Index ops such as ReorderAxes and 'Meta' ops such as MapRolesOp 
    update and propagate MKL-DNN tensor layout information. MKL-DNN conversion ops to convert
    tensors from MKL-DNN layout to a graph-visible layout are inserted in a subsequent pass.
     
    Steps for creating an op kernel
    1) Check if op is supported by MKL-DNN
    2) Marshall op parameters from graph to pass on to MKL-DNN 
    3) Create or extract MKL-DNN layouts for inputs
    4) Create MKL-DNN op kernel
    5) Remember output MKL-layout and MKL-visible axes for use by ops downstream
    
    """

    def __init__(self, mkldnn):
        assert mkldnn.mkldnn_enabled
        self.mkldnn = mkldnn

    def set_mkl_layout_data(self, op, mkl_axes):
        mkl_layout = self.mkldnn.output_layout(self.mkldnn.kernels[op.name], 0)
        if mkl_layout:
            self.mkldnn.op_layouts[op.name] = (mkl_layout, mkl_axes)

    @generic_method(dispatch_base_type=Op)
    def visit(self, op):
        pass

    @visit.on_type(BatchnormOp)
    def visit(self, op):
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
            inputs_shape = [C, H, W, N]
            mean_size = mean.axes.lengths[0]
            gamma_shape = gamma.axes.lengths[0]
            bias_shape = bias.axes.lengths[0]
            variance_size = variance.axes.lengths[0]
            outputs_shape = op.axes.lengths

            # weights is 2 dimensional, 1-st dimension contains gamma parameter, 2-nd dimension contains beta parameter.
            weights_shape = [2, gamma_shape]
            weights_shape_arg = ((ct.c_int) * len(weights_shape))(*weights_shape)

            input_shape_arg = ((ct.c_int) * len(inputs_shape))(*inputs_shape)
            outputs_shape_arg = ((ct.c_int) * len(outputs_shape))(*outputs_shape)

            (inputs_layout, mkl_axes) = get_mkl_layout(self.mkldnn, unflatten_inputs, [4, 0, 2, 3], True)
            mean_layout = None
            variance_layout = None
            
            op_id = len(self.mkldnn.kernels)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel(op_id)

            self.mkldnn.batchnorm_fprop_kernel(
                self.mkldnn.mkldnn_engine, len(inputs_shape), len(weights_shape),
                len(outputs_shape), mean_size, variance_size, input_shape_arg, weights_shape_arg, outputs_shape_arg,
                op.eps, inputs_layout, None, mean_layout, variance_layout, data_type, self.mkldnn.kernels[op.name])
            #mkl_order = [4, 0, 2, 3]
            #op_axes = Axes.as_flattened_list(op.axes)
            #out_axes = get_axes_mkl_order(op_axes, mkl_order)
            self.set_mkl_layout_data(op, mkl_axes)
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(ConvolutionOp)
    def visit(self, op):
            input = op.args[0]
            filter = op.args[1]
            # Only 2D convolution supported in MKLDNN for now
            if (op.args[0].axes.find_by_name('__NG_DEPTH').size != 1):
                return
            # Only single precision float supported for now
            if (op.dtype.type != np.float32):
                return

            data_type = self.mkldnn.datatype[op.dtype.type]
            # Assumes (C, D, H, W, N) for convolution axes 
            input_shape = get_size_mkl_order(input.axes, [4, 0, 2, 3])
            filter_shape = get_size_mkl_order(filter.axes, [4, 0, 2, 3])
            output_shape = get_size_mkl_order(op.axes, [4, 0, 2, 3])
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
            (input_layout, mkl_axes) = get_mkl_layout(self.mkldnn, input, [4, 0, 2, 3], True)
            filter_layout = None

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
            mkl_order = [4, 0, 2, 3]
            out_axes = get_axes_mkl_order(op.axes, mkl_order)
            self.set_mkl_layout_data(op, out_axes)
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(bprop_conv)
    def visit(self, op):
            input = op.args[0]
            filter = op.args[1]
            # Only 2D convolution supported in MKLDNN for now
            if (op.args[0].axes.find_by_name('__NG_DEPTH').size != 1):
                return
            # Only single precision float supported for now
            if (op.dtype.type != np.float32):
                return

            data_type = self.mkldnn.datatype[op.dtype.type]
            # Assumes (C, D, H, W, N) for convolution axes 
            input_shape = get_size_mkl_order(input.axes, [4, 0, 2, 3])
            filter_shape = get_size_mkl_order(filter.axes, [4, 0, 2, 3])
            output_shape = get_size_mkl_order(op.axes, [4, 0, 2, 3])
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
            (input_layout, mkl_axes) = get_mkl_layout(self.mkldnn, input, [4, 0, 2, 3], True)
            filter_layout = None
            
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
            mkl_order = [4, 0, 2, 3]
            out_axes = get_axes_mkl_order(op.axes, mkl_order)
            self.set_mkl_layout_data(op, out_axes)
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(update_conv)
    def visit(self, op):
            delta = op.args[0]
            inputs = op.args[1]
            # Only 2D convolution supported in MKLDNN for now
            if (delta.axes.find_by_name('__NG_DEPTH').size != 1):
                return
            # Only single precision float supported for now
            if (op.dtype.type != np.float32):
                return

            data_type = self.mkldnn.datatype[op.dtype.type]
            # Assumes (C, D, H, W, N) for convolution axes 
            delta_shape = get_size_mkl_order(delta.axes, [4, 0, 2, 3])
            filter_shape = get_size_mkl_order(op.axes, [4, 0, 2, 3])
            inputs_shape = get_size_mkl_order(inputs.axes, [4, 0, 2, 3])
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
            (delta_layout, mkl_axes) = get_mkl_layout(self.mkldnn, delta, [4, 0, 2, 3], True)
            filter_layout = None
            (inputs_layout, mkl_axes) = get_mkl_layout(self.mkldnn, inputs, [4, 0, 2, 3], True)
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
            # self.set_mkl_layout_data(op, None)
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(ReluOp)
    def visit(self, op):
            if (op.dtype.type != np.float32):
                return
            #if (len(op.axes) != 5 and len(op.axes) != 2):
            if (len(op.axes) != 5):
                return
            data_type = self.mkldnn.datatype[op.dtype.type]
            input = op.args[0]
            if len(op.axes) == 5:
              (input_layout, mkl_axes) = get_mkl_layout(self.mkldnn, input, [4, 0, 2, 3], True)
            elif len(op.axes) == 2:
              (input_layout, mkl_axes) = get_mkl_layout(self.mkldnn, input, [1, 0])
            input_size = np.prod(input.axes.lengths)
            op_id = len(self.mkldnn.kernels)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel(op_id)
            self.mkldnn.relu_fprop_kernel(
                self.mkldnn.mkldnn_engine, 
                input_size, op.slope,
                input_layout,
                data_type,
                self.mkldnn.kernels[op.name])
            mkl_order = get_order_from_axes(input.axes, mkl_axes)
            out_axes = get_axes_mkl_order(op.axes, mkl_order)
            self.set_mkl_layout_data(op, out_axes)
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(BpropReluOp)
    def visit(self, op):
            if (op.dtype.type != np.float32):
                return
            #if (len(op.axes) != 5 and len(op.axes) != 2):
            if (len(op.axes) != 5):
                return
            data_type = self.mkldnn.datatype[op.dtype.type]
            delta = op.args[0]
            fprop_src = op.args[1]
            if len(op.axes) == 5:
              (delta_layout, mkl_axes) = get_mkl_layout(self.mkldnn, delta, [4, 0, 2, 3], True)
            elif len(op.axes) == 2:
              (delta_layout, mkl_axes) = get_mkl_layout(self.mkldnn, delta, [1, 0])
            if len(op.axes) == 5:
              (fprop_src_layout, _) = get_mkl_layout(self.mkldnn, fprop_src, [4, 0, 2, 3], True)
            elif len(op.axes) == 2:
              (fprop_src_layout, _) = get_mkl_layout(self.mkldnn, fprop_src, [1, 0])
            input_size = np.prod(delta.axes.lengths)
            op_id = len(self.mkldnn.kernels)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel(op_id)
            self.mkldnn.relu_bprop_kernel(
                self.mkldnn.mkldnn_engine, 
                input_size, op.fprop.forwarded.slope,
                fprop_src_layout, delta_layout,
                data_type,
                self.mkldnn.kernels[op.name])
            mkl_order = get_order_from_axes(delta.axes, mkl_axes)
            out_axes = get_axes_mkl_order(op.axes, mkl_order)
            self.set_mkl_layout_data(op, out_axes)
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(PoolingOp)
    def visit(self, op):
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
            input_shape = get_size_mkl_order(input.axes, [4, 0, 2, 3])
            output_shape = get_size_mkl_order(op.axes, [4, 0, 2, 3])
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
            (input_layout, mkl_axes) = get_mkl_layout(self.mkldnn, input, [4, 0, 2, 3], True)

            op_id = len(self.mkldnn.kernels)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel(op_id)
            self.mkldnn.pool_fprop_kernel(
                self.mkldnn.mkldnn_engine,
                len(input_shape), len(output_shape),
                input_shape_arg, kernel_sizes, output_shape_arg,
                stride_data, pad_data, pool_type,
                input_layout, data_type, self.mkldnn.kernels[op.name])
            mkl_order = get_order_from_axes(input.axes, mkl_axes)
            out_axes = get_axes_mkl_order(op.axes, mkl_order)
            self.set_mkl_layout_data(op, out_axes)
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(BpropPoolOp)
    def visit(self, op):
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
            input_shape = get_size_mkl_order(input.axes, [4, 0, 2, 3])
            output_shape = get_size_mkl_order(op.axes, [4, 0, 2, 3])
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
            (input_layout, mkl_axes) = get_mkl_layout(self.mkldnn, input, [4, 0, 2, 3], True)

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
            mkl_order = get_order_from_axes(input.axes, mkl_axes)
            out_axes = get_axes_mkl_order(op.axes, mkl_order)
            self.set_mkl_layout_data(op, out_axes)
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(DotLowDimension)
    def visit(self, op):
            x = op.args[0]
            y = op.args[1]

            # Sanity check tensor shapes
            if (len(x.axes.lengths) != 2) or (len(y.axes.lengths) != 2):
                return
            # Only single precision float supported for now
            if op.dtype != np.float32:
                return

            #if not x.name in self.mkldnn.op_layouts:
            #  return

            x_shape = get_size_mkl_order(x.axes, [0,1])
            y_shape = get_size_mkl_order(y.axes, [1,0])
            o_shape = get_size_mkl_order(op.axes, [0,1])

            x_shape_arg = ((ct.c_int) * len(x_shape))(*x_shape)
            y_shape_arg = ((ct.c_int) * len(y_shape))(*y_shape)
            o_shape_arg = ((ct.c_int) * len(o_shape))(*o_shape)

            (x_layout, mkl_axes) = get_mkl_layout(self.mkldnn, x, [0, 1], True)
            (y_layout, _) = get_mkl_layout(self.mkldnn, y, [1, 0], False)
            data_type = self.mkldnn.datatype[op.dtype.type]

            op_id = len(self.mkldnn.kernels)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel(op_id)
            self.mkldnn.innerproduct_fprop_kernel(
                    self.mkldnn.mkldnn_engine,
                    len(x_shape), len(y_shape), 1, len(o_shape),
                    x_shape_arg, y_shape_arg, None, o_shape_arg,
                    x_layout, y_layout, None,
                    data_type, self.mkldnn.kernels[op.name])

            out_axes = get_axes_mkl_order(op.axes, [0, 1])
            self.set_mkl_layout_data(op, out_axes)
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])


    @visit.on_type(Add)
    def visit(self, op):
            return
            I_array1 = op.args[0]
            I_array2 = op.args[1]
            # Sanity check for tensor shapes
            # TODO: this check does not work - Sum does not have flags attribute.
            #if (not (I_array1.flags['C_CONTIGUOUS'] and
            #         I_array2.flags['C_CONTIGUOUS'])):
            #    return
            if (op.dtype.type != np.float32):
                return
            if len(I_array1.shape) != 1 or len(I_array2.shape) != 1:
                return

            array1_shape = I_array1.axes.lengths
            array2_shape = I_array2.axes.lengths
            out_shape = op.axes.lengths

            input1_shape = ((ct.c_int) * len(array1_shape))(*array1_shape)
            input2_shape = ((ct.c_int) * len(array2_shape))(*array2_shape)
            output_shape = ((ct.c_int) * len(out_shape))(*out_shape)

            (input1_layout, mkl_axes) = get_mkl_layout(self.mkldnn, inputs, [0])
            (input2_layout, _) = get_mkl_layout(self.mkldnn, inputs, [0])
            #(input1_layout, mkl_axes) = self.get_mkl_layout_data(I_array1, 4)
            #(input2_layout, _) = self.get_mkl_layout_data(I_array2, 4)
            data_type = self.mkldnn.datatype[op.dtype.type]

            op_id = len(self.mkldnn.kernels)
            self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel(op_id)
            self.mkldnn.add_kernel(
                    self.mkldnn.mkldnn_engine,
                    len(I_array1.shape), len(I_array2.shape), len(output_shape),
                    input1_shape, input2_shape, output_shape,
                    input1_layout, input2_layout,
                    2,
                    data_type, self.mkldnn.kernels[op.name])
            out_axes = get_axes_mkl_order(op.axes, [0])
            self.set_mkl_layout_data(op, out_axes)
            if self.mkldnn.mkldnn_verbose:
                print
                print(op_id, op.name)
                self.mkldnn.print_kernel(self.mkldnn.kernels[op.name])

    @visit.on_type(ContiguousOp)
    def visit(self, op):
        arg = op.args[0]
        arg_td = arg.tensor_description()
        op_td = op.tensor_description()
        if arg.name in self.mkldnn.op_layouts:
            self.mkldnn.op_layouts[op.name] = self.mkldnn.op_layouts[arg.name]

    @visit.on_type(MapRolesOp)
    def visit(self, op):
        arg = op.args[0]
        if arg.name in self.mkldnn.op_layouts:
          (mkl_layout, mkl_axes) =  self.mkldnn.op_layouts[arg.name]
          order = get_order_from_axes(arg.axes, mkl_axes)
          new_axes = get_axes_mkl_order(op.axes, order)
          self.mkldnn.op_layouts[op.name] = (mkl_layout, new_axes)

    @visit.on_type(ReorderAxes)
    def visit(self, op):
        arg = op.args[0]
        if arg.name in self.mkldnn.op_layouts:
            self.mkldnn.op_layouts[op.name] = self.mkldnn.op_layouts[arg.name]


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
        all_axis = Axes.as_flattened_list(op.axes)
        (mkl_layout, mkl_axes) = op.in_layout
        mkl_axes_order = get_order_from_axes(op.axes, mkl_axes)
        (out_layout, _) = get_mkl_layout(self.mkldnn, op, mkl_axes_order, True)
        ndims = len(mkl_axes)
        dims = get_size_mkl_order(op.axes, mkl_axes_order)
        dims_arg = ((ct.c_int) * ndims)(*dims)
        op_id = len(self.mkldnn.kernels)
        self.mkldnn.kernels[op.name] = self.mkldnn.create_empty_kernel(op_id)
        self.mkldnn.reorder_kernel(
          self.mkldnn.mkldnn_engine,
          ndims, dims_arg,
          self.mkldnn.datatype[op.dtype.type],
          self.mkldnn.memory_format['blocked'],
          mkl_layout, out_layout,
          self.mkldnn.kernels[op.name]
        )
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
        if op.name in self.mkldnn.kernels or op.name in self.mkldnn.op_layouts:
            # MKL Op or an MKL layout pass-through op
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

    @visit.on_type(ContiguousOp)
    def visit(self, op):
        arg = op.args[0]
        if arg.name in self.mkldnn.op_layouts:
            # Input in MKL layout.
            # Expect downstream ops to handle MKL layout or insert explicit conversions
            self.replace_op(op, op.args[0])
        elif isinstance(arg, MklReorderOp):
          # TODO(jbobba) - Can we eliminate ContiguousOp here?
          td = arg.tensor_description()
          arg_td = arg.args[0].tensor_description()
          #self.replace_op(op, op.args[0])

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
            # TODO(jbobba): Verify this case
            returns = op.returns
            op.returns = OrderedSet()
            for orig_op in returns:
                if orig_op.forwarded.name in self.mkldnn.op_layouts:
                    reorder_op = self.get_reorder_op(orig_op.forwarded)
                    op.returns.add(reorder_op)
                    op.add_control_dep(reorder_op)
                else:
                    op.returns.add(orig_op)
        else:
            pass

