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


def cast_axes_for_matmul(ng_input_left, ng_input_right):  # type: (Op, Op) -> (Op, Op)
    left, right = ng_input_left, ng_input_right
    left_num_axes = len(left.axes)
    right_num_axes = len(right.axes)

    if left_num_axes == right_num_axes == 1:
        # vector @ vector
        # cast to axes: i, icast_axes_for_matmul
        assert left.shape.lengths == right.shape.lengths, \
            "Vector lengths must be equal for multiplication."
        if left.shape != right.shape:
            right = ng.cast_axes(right, axes=left.axes)

    elif left_num_axes == 1:
        # vector @ matrix
        # cast to axes: i, ...ij
        if left.axes[0] != right.axes[-2]:
            left = ng.cast_axes(left, axes=right.axes[-2])

    elif right_num_axes == 1:
        # matrix @ vector
        # cast to axes: ...i, i
        if left.axes[-1] != right.axes[0]:
            right = ng.cast_axes(right, axes=left.axes[-1])

    else:
        # matrix @ matrix
        # cast to axes: ...ij, ...jk
        right_axes = [ng.make_axis(name='DOT_{}'.format(i), length=axis.length)
                      for i, axis in enumerate(right.shape)]
        right_axes[-2] = left.axes[-1]
        right = ng.cast_axes(right, axes=right_axes)

    return left, right
