# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

from __future__ import division
from __future__ import print_function

import numpy as np
import ngraph as ng
from ngraph.frontends.tensorflow.tf_importer.utils_pos_axes import make_pos_axes


def reorder_axes(input_tensor, input_template, output_template):
    # type: (TensorOp, str, str) -> TensorOp
    """
    Reorder input_tensor axes based on a template defined by two strings.

    Each letter of the template string denotes an axis. If a letter is not present in the
    input_axes string, but is present in output_axes, a new axis with length=1 will be added.

    E.g. `reorder_axes(input_tensor, 'NCHW', 'CDHWN')` will add an axis named D with length=1

    :param input_tensor: ngraph op, with a set of axes matching input_template
    :param input_template: string with one letter for each axis e.g. 'NCHW'
    :param output_template: string with one letter for each axis in a different order e.g. 'CHWN'
    :return: broadcast Op which reorders the axes of the input tensor
    """
    if not len(set(input_template)) == len(input_template):
        raise ValueError('Input axes names cannot repeat.')

    if not len(set(output_template)) == len(output_template):
        raise ValueError('Output axes names cannot repeat.')

    output_axes = []
    for output_axis_name in output_template:
        output_axes.append(input_tensor.axes[input_template.index(output_axis_name)]
                           if output_axis_name in input_template
                           else ng.make_axis(name=output_axis_name, length=1))

    return ng.broadcast(input_tensor, axes=output_axes)


def rename_axes(input_tensor, output_template):  # type: (TensorOp, str) -> TensorOp
    """
    Rename tensor axes according to letter names given in `output_template`.

    Example: if `output_template` is 'NHWC', then axes will be renamed to 'N', 'H', 'W' and 'C'.

    :param input_tensor: ngraph TensorOp
    :param output_template: string with one letter per axis in `input_tensor`
    :return: ngraph TensorOp with renamed axes
    """
    output_axes = [ng.make_axis(length=input_tensor.axes[i].length, name=output_template[i])
                   for i in range(len(input_tensor.axes))]
    return ng.cast_axes(input_tensor, axes=ng.make_axes(output_axes))


def reshape_workaround(data, shape_out):  # type: (TensorOp, Sequence[int]) -> TensorOp
    """Limited workaround for tensor reshape operation."""
    shape_in = data.shape.lengths

    if np.prod(shape_in) != np.prod(shape_out):
        raise ValueError('Total size of input (%d) and output (%d) dimension mismatch.',
                         np.prod(shape_in), np.prod(shape_out))

    ndims_out = len(shape_out)
    if ndims_out == 1:
        tensor = ng.flatten(data)
    elif ndims_out == 2:
        cumprods = list(np.cumprod(shape_in))
        flatten_at_idx = cumprods.index(shape_out[0]) + 1
        tensor = ng.flatten_at(data, flatten_at_idx)
    else:
        raise NotImplementedError('Reshape can only support flatten to 1d or 2d.')

    return ng.cast_axes(tensor, make_pos_axes(shape_out))
