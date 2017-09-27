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
from six import string_types
import ngraph as ng
from ngraph.op_graph.axes import Axis, IncompatibleAxesError


_SHADOW_AXIS_POSTFIX = '__NG_SHADOW'


class Namespace():
    pass


# Define the standard Neon axes
ax = Namespace()
ax.N = ng.make_axis(name='N', docstring="minibatch size")
ax.REC = ng.make_axis(name='REC', docstring="recurrent axis")
ax.Y = ng.make_axis(name="Y", docstring="target")


def shadow_axes_map(axes):
    """
    Args:
        iterable of Axis objects

    Returns:
        A map from axes to shadow axes to prevent axes from matching in
        subsequent Ops.
    """
    return {
        make_shadow_axis(axis): axis for axis in axes
    }


def make_shadow_axis(axis):
    return ng.make_axis(axis.length, name=axis.name + _SHADOW_AXIS_POSTFIX)


def is_shadow_axis(axis):
    return axis.name.endswith(_SHADOW_AXIS_POSTFIX)

def assert_no_shadow_axes(axes, variable_name='axes'):
    if any(is_shadow_axis(axis) for axis in axes):
        raise ValueError((
            "Shadow Axes are not allowed in {}. Found {}."
        ).format(
            variable_name,
            [axis for axis in axes if is_shadow_axis(axis)],
        ))

def reorder_spatial_axes(tensor, channel_axis, spatial_axes):
    """
    Reorders the axes of the input tensor in preparation for a spatial op (i.e. convolution,
    deconvolution, or pooling).

    Arguments:
        tensor (TensorOp): The input tensor whose axes must be a subset of those specified in
            channel_axis, spatial_axes and a batch axis. Missing axes in tensor will be added.
        channel_axis (Axis, str): The axis or axis name to use as the "channel" axis type
        spatial_axes (tuple of Axis or str): Tuple of axis or axis names to use as the "depth",
            "height", and "width" axis types, in that order.

    Returns:
        tensor with 5 dimensions, ordered as "channel", "depth", "height", "width", "batch"

    Raises:
        IncompatibleAxesError: The tensors' axes are incompatible with spatial ops using the
            given axis types.
    """

    if len(tensor.axes) > 5:
        raise IncompatibleAxesError("spatial ops cannot have more than 5 axes, "
                                    "found {}".format(len(tensor.axes)))

    def expand_with_name(tensor, axis, index=0):
        if isinstance(axis, Axis):
            if axis in tensor.axes:
                return tensor, axis
            if (axis.length is not None) and (axis.length > 1):
                raise IncompatibleAxesError("Cannot expand tensor to an axis with length > 1: {}"
                                            ", length={}".format(axis.name, axis.length))
            axis.length = 1
        else:
            if axis in tensor.axes.names:
                return tensor, tensor.axes.find_by_name(axis)[0]
            axis = ng.make_axis(name=axis, length=1)
        return ng.expand_dims(tensor, axis, index), axis

    def not_in(axes, ax):
        if isinstance(ax, string_types):
            return not_in(axes, ng.make_axis(name=ax))

        return ax not in axes

    batch_axis = tensor.axes.batch_axis()
    if batch_axis is None:
        raise IncompatibleAxesError('Spatial ops require a batch axis, but none were found: '
                                    '{}'.format(tensor.axes))

    if all(not_in(tensor.axes, ax) for ax in spatial_axes):
        raise IncompatibleAxesError("spatial_axes provided were {}, but none were found in the "
                                    "tensor: {}. All spatial ops require at least one spatial "
                                    "dimension.".format(spatial_axes, tensor.axes))

    tensor, channel_axis = expand_with_name(tensor, channel_axis)

    spatial_axes = list(spatial_axes)
    for ii, ax in enumerate(spatial_axes):
        tensor, ax = expand_with_name(tensor, ax)
        spatial_axes[ii] = ax

    new_axes = channel_axis + ng.make_axes(spatial_axes) + batch_axis
    if tensor.axes.is_not_equal_set(new_axes):
        raise IncompatibleAxesError("Found extra axes: "
                                    "{}".format(set(tensor.axes).difference(set(new_axes))))

    return ng.axes_with_order(tensor, new_axes)
