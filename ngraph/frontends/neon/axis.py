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
_WIDTH = '__NG_WIDTH'
_DEPTH = '__NG_DEPTH'


class Namespace():
    pass


# Define the standard Neon axes
ax = Namespace()
ax.N = ng.make_axis(name='N', docstring="minibatch size")
ax.REC = ng.make_axis(name='R', docstring="recurrent axis")
ax.Y = ng.make_axis(docstring="target")


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
    spatial_axes = tensor.axes.spatial_axes()
    batch_axes = tensor.axes.batch_axes()

    if len(spatial_axes) == 0 or len(spatial_axes) > 3:
        raise ValueError(
            'spatial ops can only operate on tensors with 1, 2, or 3 spatial axes.'
            'Found {}'.format(spatial_axes)
        )

    if not batch_axes:
        raise ValueError(
            'spatial ops require a batch axis'
        )

    if not tensor.axes.channel_axis():
        c = ng.make_axis(length=1, name='C')
        tensor = ng.expand_dims(tensor, c, 0)
    channel_axes = ng.make_axes(tensor.axes.channel_axis())

    if len(spatial_axes) == 1:
        w = ng.make_axis(length=1, name=_WIDTH)
        tensor = ng.expand_dims(tensor, w, 0)
        spatial_axes = spatial_axes + w

    if len(spatial_axes) == 2:
        d = ng.make_axis(length=1, name=_DEPTH)
        tensor = ng.expand_dims(tensor, d, 0)
        spatial_axes = ng.make_axes([d]) + spatial_axes

    new_axes = channel_axes + spatial_axes + batch_axes
    return ng.axes_with_order(tensor, new_axes)
