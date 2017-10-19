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

from __future__ import print_function
from __future__ import division

import logging
import ngraph as ng

from ngraph.frontends.tensorflow.tf_importer.utils_broadcast import is_compatible_broadcast_shape

logger = logging.getLogger(__name__)


def verify_axes_binary_broadcast_compatible(onnx_node, ng_inputs):
    # type: (NodeWrapper, List[TensorOp]) -> bool
    """
    Check if two ngraph input nodes have shapes compatible for an element-wise binary operation.
    """
    shape_left = tuple(axis.length for axis in ng_inputs[0].shape)
    shape_right = tuple(axis.length for axis in ng_inputs[1].shape)
    dimensions_identical = shape_left == shape_right

    broadcast_attribute = onnx_node.get_attribute('broadcast')
    broadcast_flag_set = broadcast_attribute and broadcast_attribute.get_value() == 1
    if not dimensions_identical and not broadcast_flag_set:
        logger.warning('%s node (%s): operands have different dimensions, and "broadcast"'
                       ' attribute is not set. ', onnx_node.op_type, onnx_node.name)

    axis_attribute = onnx_node.get_attribute('axis')
    if axis_attribute:
        raise NotImplementedError('%s node (%s): "axis" attribute not supported yet.',
                                  onnx_node.op_type, onnx_node.name)

    if not dimensions_identical and not is_compatible_broadcast_shape(shape_right, shape_left):
        logger.error('%s node (%s): operands have shapes incompatible for broadcasting.',
                     onnx_node.op_type, onnx_node.name)
        return False
    return True


def get_reduction_axes(onnx_node):  # type: (NodeWrapper) -> Axes
    """
    Create an ngraph Axes object for a subset of axes to be used in a reduction operation.
    """
    input_tensor = onnx_node.get_ng_inputs()[0]
    axes_attribute = onnx_node.get_attribute('axes')

    if axes_attribute is None:
        ng_reduction_axes = input_tensor.axes
    else:
        ng_reduction_axes = ng.make_axes([input_tensor.axes[ind] for ind in
                                          axes_attribute.get_value()])

    return ng_reduction_axes


def make_reduction_op(ng_op_type, onnx_node, ng_input):
    # type: (Callable, NodeWrapper, TensorOp) -> Op
    """
    Create an ngraph Op node for a reduction operation (min, max, sum, etc.)

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
