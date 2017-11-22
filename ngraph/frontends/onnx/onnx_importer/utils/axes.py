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
