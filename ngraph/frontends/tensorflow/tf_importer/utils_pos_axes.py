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

"""
Utilities related to positional axes

Idea:
    Each tensor is represented using axes name starting with 'POS_' with number
    indicating the axes index counted from the right.

Example: a 4d Tensor
    ('POS_3', 'POS_2', 'POS_1', 'POS_0')

In order to support:
       (5, 1, 4, 1)
    +     (3, 1, 2)
    ---------------
       (5, 3, 4, 2)

We do broadcast first:
    (5, 1, 4, 1)  ->    (5, 3, 4, 2)
       (3, 1, 2)  ->  + (5, 3, 4, 2)
                     ---------------
                        (5, 3, 4, 2)
"""
import ngraph as ng

POS_AXIS_PREFIX = 'POS_'


def make_pos_axis(length, pos, prefix=POS_AXIS_PREFIX):
    """
    Make positional Axis of length `length` and of position `pos`

    Args:
        length: the length of the Axis, can be None
        pos: position of the Axis

    Example:
        for a tensor of shape (4, 3), it's positional axes are:
        [make_pos_axis(4, 1), make_pos_axis(3, 1)]

    Returns:
        Axis object
    """
    return ng.make_axis(length, name='%s%s' % (prefix, pos))


def make_pos_axes(shape, positions=None, prefix=POS_AXIS_PREFIX):
    """
    Convert shape to axes.

    Arguments:
        shape: Shape of tensor
        positions: List of positions. If positions is None, then the default
                   tensor positions is used. For example, by default, a 4d
                   tensor would have positions (3, 2, 1, 0).

    Returns:
        Axes: Axes for shape.
    """
    if positions is None:
        positions = reversed(list(range(len(shape))))
    else:
        # there should be no duplicates in positions
        if len(set(positions)) != len(positions):
            raise ValueError(
                "There are duplicated positions in {}".format(positions))
        # the length of shape and positions should be the same
        if len(shape) != len(positions):
            raise ValueError("The length of shape {} should be the same as the"
                             "length of positions {}".format(len(shape),
                                                             len(positions)))
    if not shape:
        return ng.make_axes()
    axes = [make_pos_axis(length, pos, prefix=prefix)
            for length, pos in zip(shape, positions)]
    return axes


def cast_to_pos_axes(x, prefix=POS_AXIS_PREFIX):
    """
    Cast an op to positional axes.

    E.g.
    before: x.axes == ['H', 'W']
    after:  x.axes == ['pos_1', 'pos_0']

    Args:
        x: ngraph op

    Returns:
        x casted to positional axes
    """
    return ng.cast_axes(x, make_pos_axes(x.axes.lengths, prefix=prefix))


def reorder_pos_axes(x, prefix=POS_AXIS_PREFIX):
    """
    Reorder x's axes to descending positional axes. E.g.
    x's axes: [POS_1, POS_2, POS_0] => [POS_2, POS_1, POS_0]

    Args:
        x: ngrpah op

    Returns:
        x reordered to descending positional axes.
    """
    # get axes names
    axes_names = [axis.name for axis in x.axes]
    num_axes = len(axes_names)

    # check axes names are valid
    for name in axes_names:
        if name[:len(prefix)] != prefix:
            raise ValueError("axis {} is not a valid positional axes, "
                             "to be valid, must have prefix {}"
                             .format(name, prefix))

    axes_positions = [int(name[len(prefix):]) for name in axes_names]
    if sorted(axes_positions) != list(range(num_axes)):
        raise ValueError("axes positions {} must be continuous integers "
                         "starting from 0")

    # special case, x is already in a good order
    if (axes_positions == reversed(list(range(num_axes)))):
        return x

    # get a position -> length map
    map_pos_length = dict()
    for pos, length in zip(axes_positions, x.axes.lengths):
        map_pos_length[pos] = length

    # get shape after reordering
    new_shapes = [map_pos_length[pos] for pos in reversed(list(range(num_axes)))]

    return ng.axes_with_order(x, axes=make_pos_axes(new_shapes))
