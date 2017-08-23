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
import ngraph as ng


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


def reorder_spatial_axes(tensor):
    """
    Assumes we are getting a C, H, N, or C, H, W, N, or C, D, H, W, N
    """

    def expand_with_role(tensor, role, index=0):
        axis = ng.make_role_axis(role, length=1)
        return ng.expand_dims(tensor, axis, index), axis

    channel_axis = tensor.axes.channel_axis()
    spatial_roles = ("depth", "height", "width")
    spatial_axes = [tensor.axes.role_axis(role) for role in spatial_roles]
    batch_axis = tensor.axes.batch_axis()
    role_axes = set([ax for ax in spatial_axes + [channel_axis, batch_axis] if ax is not None])
    diff_axes = role_axes.difference(set(tensor.axes))
    if len(diff_axes) > 0:
        raise ValueError("Found extra axes: {}".format(list(diff_axes)))

    if batch_axis is None:
        raise ValueError('spatial ops require a batch axis')

    if all((ax is None) for ax in spatial_axes):
        raise ValueError("spatial ops require at least one spatial axis, found none")

    if channel_axis is None:
        tensor, channel_axis = expand_with_role(tensor, "channel")

    for ii, role in enumerate(spatial_roles):
        ax = spatial_axes[ii]
        if ax is None:
            tensor, ax = expand_with_role(tensor, role)
            spatial_axes[ii] = ax

    new_axes = channel_axis + spatial_axes + batch_axis
    return ng.axes_with_order(tensor, new_axes)
