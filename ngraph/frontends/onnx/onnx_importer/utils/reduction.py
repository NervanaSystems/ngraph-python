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


def get_reduction_axes(onnx_node):  # type: (NodeWrapper) -> Axes
    """Create an ngraph Axes object for a subset of axes to be used in a reduction operation."""
    input_tensor = onnx_node.get_ng_inputs()[0]
    attribute_axes = onnx_node.get_attribute_value('axes')
    attribute_axis = onnx_node.get_attribute_value('axis')

    if attribute_axes is not None:
        ng_reduction_axes = ng.make_axes([input_tensor.axes[ind] for ind in attribute_axes])
    elif attribute_axis is not None:
        ng_reduction_axes = ng.make_axes((input_tensor.axes[attribute_axis],))
    else:
        ng_reduction_axes = input_tensor.axes

    return ng_reduction_axes


def make_reduction_op(ng_op_type, onnx_node, ng_input):
    # type: (Callable, NodeWrapper, TensorOp) -> Op
    """
    Create an ngraph Op node for a reduction operation (min, max, sum, etc.).

    :param ng_op_type: an ngraph reduction factory function such as ng.max, etc.
    :param onnx_node: wrapped ONNX node
    :param ng_input: ngraph Op to be used as input to the reduction node
    """
    reduction_ng_axes = get_reduction_axes(onnx_node)
    op = ng_op_type(ng_input, reduction_axes=reduction_ng_axes)

    if onnx_node.get_attribute_value('keepdims', default=1):
        for axis in reduction_ng_axes:
            pos = ng_input.axes.index(axis)
            new_axis = ng.make_axis(length=1, name=axis.name)
            op = ng.expand_dims(op, new_axis, pos)

    return op
