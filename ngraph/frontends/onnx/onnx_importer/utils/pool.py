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

from __future__ import division
from __future__ import print_function

import ngraph as ng
from ngraph.frontends.onnx.onnx_importer.utils.axes import reorder_axes
from ngraph.frontends.onnx.onnx_importer.utils.misc import verify_symmetric_padding


def get_pool_params(onnx_node):  # type: (NodeWrapper) -> Dict
    """
    Parse ONNX pooling operation attributes and produce an ngraph compatible pool_params dict

    :param onnx_node: wrapped ONNX node for a pooling operation op
    :return: dict of pool_params for ng.pooling
    """
    pads = onnx_node.get_attribute_value('pads')  # Padding along each axis
    strides = onnx_node.get_attribute_value('strides')  # stride along each axis
    kernel_shape = onnx_node.get_attribute_value('kernel_shape')

    verify_symmetric_padding(onnx_node)

    pad_c, pad_h, pad_w, pad_d = 0, 0, 0, 0
    if pads and len(pads) == 4:  # ONNX input axes NCHW
        pad_h, _, pad_w, _ = pads
    elif pads and len(pads) == 6:  # ONNX input axes NCHWD
        pad_h, _, pad_w, _, pad_d, _ = pads

    str_c, str_h, str_w, str_d = 1, 1, 1, 1
    if strides and len(strides) == 2:  # ONNX input axes order NCHW
        str_h, str_w = strides
    elif strides and len(strides) == 3:  # ONNX input axes order NCHWD
        str_h, str_w, str_d = strides

    ker_c, ker_h, ker_w, ker_d = 1, 1, 1, 1
    if kernel_shape and len(kernel_shape) == 2:  # ONNX input axes order NCHW
        ker_h, ker_w = kernel_shape
    elif kernel_shape and len(kernel_shape) == 3:  # ONNX input axes order NCHWD
        ker_h, ker_w, ker_d = kernel_shape

    if onnx_node.op_type in ['AveragePool', 'GlobalAveragePool']:
        pooling_op = 'avg'
    elif onnx_node.op_type in ['MaxPool', 'GlobalMaxPool']:
        pooling_op = 'max'
    else:
        raise NotImplementedError('%s node (%s): Unsupported pooling type.',
                                  onnx_node.op_type, onnx_node.name)

    return dict(pad_d=pad_d, pad_h=pad_h, pad_w=pad_w, pad_c=pad_c,
                str_d=str_d, str_h=str_h, str_w=str_w, str_c=str_c,
                J=ker_c, R=ker_h, S=ker_w, T=ker_d, op=pooling_op)


def make_pool_output_axes(input, pool_params):
    # type: (TensorOp, Dict) -> Axes
    """
    Prepare axes for the output of an ng.convolution operation

    :param input: ngraph tensor with pooling input data
    :param pool_params: dict of pool_params for ng.pooling
    :return: ngraph Axes compatible with pooling operation
    """
    number_output_channels = input.axes[0].length
    mini_batch_size = input.axes[-1].length

    input_d, input_h, input_w = input.axes.lengths[1:4]  # axes order C, D, H, W, N

    params = pool_params
    output_d = int((input_d + 2 * params['pad_d'] - params['T']) / params['str_d']) + 1
    output_h = int((input_h + 2 * params['pad_h'] - params['R']) / params['str_h']) + 1
    output_w = int((input_w + 2 * params['pad_w'] - params['S']) / params['str_w']) + 1

    output_axes = ng.make_axes(axes=(
        ng.make_axis(name='C', docstring='channels', length=int(number_output_channels)),
        ng.make_axis(name='D', docstring='depth', length=int(output_d)),
        ng.make_axis(name='H', docstring='height', length=int(output_h)),
        ng.make_axis(name='W', docstring='width', length=int(output_w)),
        ng.make_axis(name='N', docstring='mini-batch size', length=int(mini_batch_size)),
    ))
    return output_axes


def make_pooling_op(onnx_node, ng_inputs, custom_pool_params=None):
    # type: (NodeWrapper, List[TensorOp], Dict) -> Op
    """
    Create an ngraph pooling Op based on an ONNX node.

    :param onnx_node: wrapped ONNX node for a pooling op
    :param ng_inputs: ngraph TensorOp input tensors
    :param custom_pool_params: optional pool_params overriding values based on onnx_node
    :return: ngraph pooling op
    """
    x = ng_inputs[0]

    if len(x.axes) == 4:  # 2D pooling
        # Reshape x axes from ONNX (N, C, H, W) to ngraph (C, D, H, W, N)
        x = reorder_axes(x, 'NCHW', 'CDHWN')
    elif len(x.axes) == 5:  # 3D pooling
        # Reshape x axes from ONNX (N, C, H, W, D) to ngraph (C, D, H, W, N)
        x = reorder_axes(x, 'NCHWD', 'CDHWN')
    else:
        raise NotImplementedError('%s node (%s): only 2D and 3D pooling ops are supported.',
                                  onnx_node.op_type, onnx_node.name)

    pool_params = get_pool_params(onnx_node)
    if custom_pool_params:
        pool_params.update(custom_pool_params)

    output_axes = make_pool_output_axes(x, pool_params)

    ng_op = ng.pooling(pool_params, x, output_axes)

    # ONNX output should have axes in the order N, C, H, W, D
    ng_op = reorder_axes(ng_op, 'CDHWN', 'NCHWD')

    if len(ng_inputs[0].axes) == 4:  # 2D convolution, slice away the D axis from output
        ng_op = ng.tensor_slice(ng_op, [slice(None), slice(None), slice(None), slice(None), 0])

    return ng_op


def make_global_pooling_op(onnx_node, ng_inputs):
    # type: (NodeWrapper, List[TensorOp], Dict) -> Op
    """
    Create a ngraph global pooling operation, equivalent to pooling with kernel size equal
    to the spatial dimension of input tensor.

    :param onnx_node: wrapped ONNX node for a pooling op
    :param ng_inputs: ngraph TensorOp input tensors
    :return: ngraph pooling op
    """
    x = ng_inputs[0]

    if len(x.axes) == 4:  # ONNX input axes order NCHW
        _, _, kernel_h, kernel_w = x.axes.lengths
        pool_params = dict(R=kernel_h, S=kernel_w)
    elif len(x.axes) == 5:  # ONNX input axes order NCHWD
        _, _, kernel_h, kernel_w, kernel_d = x.axes.lengths
        pool_params = dict(R=kernel_h, S=kernel_w, T=kernel_d)
    else:
        raise NotImplementedError('%s node (%s): only 2D and 3D pooling ops are supported.',
                                  onnx_node.op_type, onnx_node.name)

    return make_pooling_op(onnx_node, ng_inputs, custom_pool_params=pool_params)
