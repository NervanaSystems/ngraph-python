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

from math import floor, ceil

import ngraph as ng
from ngraph.frontends.onnx.onnx_importer.utils.axes import reorder_axes
from ngraph.frontends.onnx.onnx_importer.utils.misc import verify_symmetric_padding
from ngraph.frontends.tensorflow.tf_importer.utils_pos_axes import cast_to_pos_axes


def get_pads(onnx_node):  # type: (NodeWrapper) -> Tuple[int, int, int]
    """
    Get padding values for the operation described by an ONNX node.

    If `auto_pad` attribute is specified as SAME_UPPER or SAME_LOWER, or VALID values are
    calculated. Otherwise values are taken from the `pads` attribute.

    `pads` value should follow [x1_begin, x2_begin..., x1_end, x2_end,...]

    :param onnx_node: wrapped ONNX node for Conv or Pool operation
    :return: tuple of numbers of pixels to pad (height, width, depth)
    """
    auto_pad = onnx_node.get_attribute_value('auto_pad')
    pads = onnx_node.get_attribute_value('pads', ())  # Padding along each axis
    kernel_shape = onnx_node.get_attribute_value('kernel_shape')

    # Attribute 'auto_pad' is deprecated, but is currently used by CNTK
    if auto_pad:
        if auto_pad == 'VALID':
            pads = [0, 0] * len(kernel_shape)

        else:
            # SAME_UPPER or SAME_LOWER mean pad the input so that the output size match the input.
            # In case of odd number add the extra padding at the end for SAME_UPPER and at the
            # beginning for SAME_LOWER.
            def pad_value(kernel_dim):  # type: (int) -> float
                return (kernel_dim - 1.0) / 2.0

            pads_starts = [floor(pad_value(dim)) if auto_pad == 'SAME_UPPER' else
                           ceil(pad_value(dim)) for dim in kernel_shape]
            pads_ends = [ceil(pad_value(dim)) if auto_pad == 'SAME_UPPER' else
                         floor(pad_value(dim)) for dim in kernel_shape]
            pads = pads_starts + pads_ends

    verify_symmetric_padding(onnx_node, pads)

    pad_h, pad_w, pad_d = 0, 0, 0
    if pads and len(pads) == 2:  # ONNX input axes NCHW
        pad_h, pad_w = pads
    if pads and len(pads) == 3:  # ONNX input axes NCHWD
        pad_h, pad_w, pad_d = pads
    if pads and len(pads) == 4:  # ONNX input axes NCHW
        pad_h, pad_w, _, _ = pads
    elif pads and len(pads) == 6:  # ONNX input axes NCHWD
        pad_h, pad_w, pad_d, _, _, _ = pads

    return pad_h, pad_w, pad_d


def get_strides(onnx_node):  # type: (NodeWrapper) -> Tuple[int, int, int]
    """
    Get number of pixels to stride operation by in each direction.

    :param onnx_node: wrapped ONNX node for Conv or Pool operation
    :return: tuple of numbers of pixels to stride by (height, width, depth)
    """
    str_h, str_w, str_d = 1, 1, 1  # default values
    strides = onnx_node.get_attribute_value('strides', ())  # stride along each axis

    if len(strides) == 2:  # ONNX input axes order NCHW
        str_h, str_w = strides
    elif len(strides) == 3:  # ONNX input axes order NCHWD
        str_h, str_w, str_d = strides

    return str_h, str_w, str_d


def get_dilations(onnx_node):  # type: (NodeWrapper) -> Tuple[int, int, int]
    """
    Get number of pixels for filter dilation in each direction.

    :param onnx_node: wrapped ONNX node for Conv or Pool operation
    :return: tuple of numbers of pixels for filter dilation (height, width, depth)
    """
    dil_h, dil_w, dil_d = 1, 1, 1  # default values
    dilations = onnx_node.get_attribute_value('dilations', ())  # dilation along each filter axis

    if len(dilations) == 2:  # ONNX input axes order NCHW
        dil_h, dil_w = dilations
    elif len(dilations) == 3:  # ONNX input axes order NCHWD
        dil_h, dil_w, dil_d = dilations

    return dil_h, dil_w, dil_d


def get_conv_params(onnx_node):  # type: (NodeWrapper) -> Dict
    """
    Parse ONNX Conv operation attributes and produce an ngraph compatible conv_params dict.

    :param onnx_node: wrapped ONNX node for Conv or ConvTranspose operation
    :return: dict of conv_params for ng.convolution
    """
    pad_h, pad_w, pad_d = get_pads(onnx_node)
    str_h, str_w, str_d = get_strides(onnx_node)
    dil_h, dil_w, dil_d = get_dilations(onnx_node)

    return {'pad_d': pad_d, 'pad_h': pad_h, 'pad_w': pad_w,
            'str_d': str_d, 'str_h': str_h, 'str_w': str_w,
            'dil_d': dil_d, 'dil_h': dil_h, 'dil_w': dil_w}


def make_conv_output_axes(input, filter, conv_params):
    # type: (TensorOp, TensorOp, Dict) -> Axes
    """
    Prepare axes for the output of an ng.convolution operation.

    :param input: ngraph tensor with convolution input data
    :param filter: ngraph tensor with convolution filter data
    :param conv_params: dict of conv_params for ng.convolution
    :return: ngraph Axes compatible with convolution operation
    """
    number_output_features = filter.axes[-1].length
    mini_batch_size = input.axes[-1].length

    input_d, input_h, input_w = input.axes.lengths[1:4]  # axes order C, D, H, W, N
    filter_d, filter_h, filter_w = filter.axes.lengths[1:4]  # axes order J, T(d), R(h), S(w), K

    def output_dim(input_x, filter_x, pad_x, str_x, dil_x):
        return floor((input_x + 2 * pad_x - filter_x - (filter_x - 1) * (dil_x - 1)) / str_x) + 1

    convp = conv_params
    output_d = output_dim(input_d, filter_d, convp['pad_d'], convp['str_d'], convp['dil_d'])
    output_h = output_dim(input_h, filter_h, convp['pad_h'], convp['str_h'], convp['dil_h'])
    output_w = output_dim(input_w, filter_w, convp['pad_w'], convp['str_w'], convp['dil_w'])

    output_axes = ng.make_axes(axes=(
        ng.make_axis(name='C', docstring='output features', length=int(number_output_features)),
        ng.make_axis(name='D', docstring='depth', length=int(output_d)),
        ng.make_axis(name='H', docstring='height', length=int(output_h)),
        ng.make_axis(name='W', docstring='width', length=int(output_w)),
        ng.make_axis(name='N', docstring='mini-batch size', length=int(mini_batch_size)),
    ))
    return output_axes


def make_convolution_op(onnx_node, ng_inputs, transpose=False):
    # type: (NodeWrapper, List[TensorOp], bool) -> Op
    """
    Create an ngraph convolution or deconvolution Op based on an ONNX node.

    :param onnx_node: wrapped ONNX node for Conv of ConvTranspose op
    :param ng_inputs: ngraph TensorOp input tensors
    :param transpose: should this be a transposed convolution?
    :return: ngraph Op for convolution or deconvolution
    """
    if len(ng_inputs) == 3:
        x, weights, bias = ng_inputs
    elif len(ng_inputs) == 2:
        x, weights = ng_inputs
        bias = ng.constant(0)
    else:
        raise ValueError('Conv node (%s): unexpected number of input values: %d.',
                         onnx_node.name, len(ng_inputs))

    # Reorder x axes from ONNX convention (N, C, H, W, D) to ngraph (C, D, H, W, N)
    # Reorder weights axes from ONNX (K, J, R, S, T) to ngraph (J, T, R, S, K)
    # Axis names follow https://ngraph.nervanasys.com/index.html/axes.html
    if len(x.axes) == 4:  # 2D convolution
        x = reorder_axes(x, 'NCHW', 'CDHWN')
        weights = reorder_axes(weights, 'KJRS', 'JTRSK')
    elif len(x.axes) == 5:  # 3D convolution
        x = reorder_axes(x, 'NCHWD', 'CDHWN')
        weights = reorder_axes(weights, 'KJRST', 'JTRSK')
    else:
        raise NotImplementedError('Conv node (%s): only 2D and 3D convolutions are supported.',
                                  onnx_node.name)

    groups = onnx_node.get_attribute_value('group', 1)
    if groups != 1:
        raise NotImplementedError('Conv node (%s): `group` attribute value %d not supported.',
                                  onnx_node.name, groups)

    # Prepare ngraph convolution operation
    conv_params = get_conv_params(onnx_node)
    output_axes = make_conv_output_axes(x, weights, conv_params)

    if transpose:
        conv = ng.deconvolution(conv_params, x, weights, axes=output_axes)

    else:
        conv = ng.convolution(conv_params, x, weights, axes=output_axes)

    conv = cast_to_pos_axes(conv) + bias

    # ONNX output should have axes in the order N, C, H, W, D
    conv = reorder_axes(conv, 'CDHWN', 'NCHWD')

    if len(ng_inputs[0].axes) == 4:  # 2D convolution, slice away the D axis from output
        conv = ng.tensor_slice(conv, [slice(None), slice(None), slice(None), slice(None), 0])

    return conv
