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
from __future__ import division

import uuid
import collections
import operator
import itertools
from functools import reduce, wraps
from frozendict import frozendict

import numpy as np
import types
from builtins import object, map, zip

from ngraph.util.names import NameableValue
from ngraph.flex.base import Flex


def default_dtype(dtype=None):
    if dtype is None:
        dtype = np.dtype(np.float32)
    elif not isinstance(dtype, Flex) and not isinstance(dtype, np.dtype):
        try:
            dtype = np.dtype(dtype)
        except TypeError:
            raise TypeError("Could not cast {} to np.dtype".format(dtype))
    return dtype


def default_int_dtype(dtype=None):
    if dtype is None:
        dtype = np.dtype(np.int32)
    elif not isinstance(dtype, Flex) and not isinstance(dtype, np.dtype):
        try:
            dtype = np.dtype(dtype)
        except TypeError:
            raise TypeError("Could not cast {} to np.dtype".format(dtype))
    return dtype


def make_axis(length=None, name=None,
              docstring=None):
    """
    Returns a new Axis.

    Args:
        length (int, optional): Length of the axis.
        name (String, optional): Name of the axis.
        batch (bool, optional): This is a batch axis. Defaults to False.
        recurrent (bool, optional): This is a recurrent axis. Defaults to False.
        docstring (String, optional): A docstring for the axis.

    Returns:
        Axis: A new Axis.
    """
    return Axis(length=length, name=name, docstring=docstring)


def make_axes(axes=()):
    """
    Makes an Axes object.

    Args:
        axes: A list of Axis.

    Returns:
        Axes: An Axes.
    """
    return Axes(axes=axes)


class Axis(object):
    """
    An Axis labels a dimension of a tensor. The op-graph uses
    the identity of Axis objects to pair and specify dimensions in
    symbolic expressions. This system has several advantages over
    using the length and position of the axis as in other frameworks:

    1) Convenience. The dimensions of tensors, which may be nested
    deep in a computation graph, can be specified without having to
    calculate their lengths.

    2) Safety. Axis labels are analogous to types in general-purpose
    programming languages, allowing objects to interact only when
    they are permitted to do so in advance. In symbolic computation,
    this prevents interference between axes that happen to have the
    same lengths but are logically distinct, e.g. if the number of
    training examples and the number of input features are both 50.

    TODO: Please add to the list...

    Arguments:
        length: The length of the axis.
        batch: Whether the axis is a batch axis.
        recurrent: Whether the axis is a recurrent axis.
    """
    __name_counter = 0

    def __init__(self,
                 length=None,
                 name=None,
                 **kwargs):
        assert 'batch' not in kwargs
        assert 'recurrent' not in kwargs

        if name is None:
            # generate name for axis if None was provided
            name = '%s_%s' % (type(self).__name__, type(self).__name_counter)
            type(self).__name_counter += 1

        self.name = name

        if length is not None and length < 0:
            raise ValueError("Axis length {} must be >= 0".format(length))
        self.__length = length

        self.uuid = uuid.uuid4()

    def named(self, name):
        self.name = name
        return self

    @property
    def is_flattened(self):
        """
        Returns:
            True if this is a flattened axis.
        """
        return False

    @property
    def is_batch(self):
        """
        Tests if an axis is a batch axis.

        Returns:
            bool: True if the axis is a batch axis.

        """
        return self.name == 'N'

    @property
    def is_recurrent(self):
        """
        Tests if an axis is a recurrent axis.

        Returns:
            bool: True if the axis is a recurrent axis.

        """
        return self.name == 'REC'

    @property
    def is_channel(self):
        """
        Tests if an axis is a channel axis.

        Returns:
            bool: True if the axis is a channel axis.

        """
        return self.name == 'C'

    @property
    def length(self):
        """
        Returns:
            The length of the axis.
        """
        return self.__length

    @length.setter
    def length(self, value):
        if value < 0:
            raise ValueError("Axis length {} must be >= 0".format(value))
        self.__length = value

    @property
    def axes(self):
        return Axes([self])

    def __repr__(self):
        return 'Axis({name}: {length})'.format(name=self.name, length=self.length)

    def __str__(self):
        return '{name}: {length}'.format(name=self.name, length=self.length)

    def __eq__(self, other):
        return isinstance(other, Axis) and self.name == other.name

    def __hash__(self):
        return hash((self.name, self.length))


def _sliced_length(s, incoming_length):
    start, stop, step = s.indices(incoming_length)

    # max with 0 so we dont ever return a negative length.  This
    # matches how python handles it internally.  Raising an exception
    # might also be reasonable.
    if step == 1:
        return max(stop - start, 0)
    elif step == -1:
        return max(start - stop, 0)
    else:
        _validate_slice(s)


def _validate_slice(s):
    if s.step not in (-1, 1, None):
        raise ValueError((
            'SlicedAxis cant currently handle a step size other '
            'than -1, 1 or None.  Was given {step} in slice {slice}'
        ).format(
            step=s.step,
            slice=s,
        ))


def slice_axis(axis, s):
    """
    Slice an axis, return complete new axis
    TODO: deprecate this after the axis refactoring

    Arguments:
        axis: the axis to be sliced
        s: slice

    Returns:
        Axis instance, the new sliced axis
    """
    # validate
    _validate_slice(s)

    # get sliced length
    new_length = None if axis.length is None else _sliced_length(s, axis.length)

    # create sliced axis
    new_axis = make_axis(length=new_length,
                         name=axis.name)
    return new_axis


def duplicates(arr):
    """
    Returns a list of Axis objects which have duplicate names in arr

    Arguments:
        arr: The iterable of Axis objects to check for duplicates in.

    Returns:
        list of Axis: duplicate Axis found in arr
    """
    # group axes by name
    axes_by_name = collections.defaultdict(list)
    for x in arr:
        axes_by_name[x.name].append(x)

    # find all names which are used by more than 1 axis, and add those axes to
    # the list of duplicates
    duplicates = []
    for name, axes in axes_by_name.items():
        if len(axes) > 1:
            duplicates.extend(axes)

    return duplicates


def with_args_as_axes(f):
    """
    A decorator to cast arguments to axes.

    Arguments:
        f: The function to be decorated.

    Returns:
        The decorated function.
    """
    @wraps(f)
    def wrapper(*args):
        """
        The decorated function. Performs the conversion
        to Axes.

        Arguments:
          *args: Arguments intended for the original function.

        Returns:
            Return value of the original function.
        """
        args = [Axes(arg) for arg in args]
        return f(*args)
    return wrapper


class Axes(object):
    """
    An Axes is a tuple of Axis objects used as a label for a tensor's
    dimensions.
    """

    def __init__(self, axes=None):
        if axes is None:
            axes = []
        elif isinstance(axes, Axis):
            axes = [axes]
        elif isinstance(axes, types.GeneratorType):
            axes = tuple(axes)
        elif isinstance(axes, (list, tuple)) and not isinstance(axes, Axes):
            axes = tuple(axes)

        def convert(seq):
            """
            Converts the sequence and all nested sequences in it into Axis and
            Axes objects.

            Arguments:
                seq: The sequence to convert.

            Returns:
                The converted axes.
            """
            elems = []
            for x in seq:
                if isinstance(x, collections.Iterable):
                    x = Axes(convert(x)).flatten()
                elems.append(x)
            return elems

        axes = convert(axes)

        for x in axes:
            if not isinstance(x, Axis):
                raise ValueError((
                    'tried to initialize an Axes with object type '
                    '{found_type}.  all values should be an instance '
                    'of a type which inherits from Axis.'
                ).format(
                    found_type=type(x),
                ))

        if duplicates(axes):
            raise ValueError(
                'The axes labels of a tensor cannot contain duplicates.  Found: {}'
                .format(str(duplicates(axes)))
            )
        self._axes = tuple(axes)
        self.uuid = uuid.uuid4()

    @property
    def full_lengths(self):
        """
        Returns all information about the lengths of the axis objects
        in this Axes in the form of a nested tuple. An element of the
        outer tuple that is itself a tuple contains the restored lengths
        of axes that have been flattened in this Axis object.

        Returns:
            tuple: A nested tuple with the axis lengths.
        """
        return tuple(x.axes.full_lengths if x.is_flattened
                     else x.length for x in self)

    @property
    def names(self):
        """
        Returns:
            tuple: The names of the outer axes.
        """
        return tuple(x.name for x in self)

    @property
    def lengths(self):
        """
        Returns:
            tuple: The lengths of the outer axes.
        """
        return tuple(x.length for x in self)

    def batch_axes(self):
        """
        Returns:
            The tensor's batch Axis wrapped in an Axes object if there is one
            on this tensor, otherwise returns None
        """
        batch_axis = self.batch_axis()
        if batch_axis:
            return Axes([batch_axis])
        else:
            return None

    def batch_axis(self):
        """
        Returns:
            The tensor's batch Axis or None if there isn't one.
        """
        for axis in self:
            if axis.is_batch:
                return axis

    def channel_axis(self):
        """
        Returns:
            The tensor's batch Axis or None if there isn't one.
        """
        for axis in self:
            if axis.is_channel:
                return axis

    def spatial_axes(self):
        """
        Returns:
            The Axes subset that are not batch, recurrent, or channel axes.
        """
        return self.feature_axes() - self.channel_axis()

    def sample_axes(self):
        """
        Returns:
            The Axes subset that are not batch axes.
        """
        return Axes(axis for axis in self if not axis.is_batch)

    def feature_axes(self):
        """
        Returns:
            The Axes subset that are not batch or recurrent axes.
        """
        return Axes(axis for axis in self if not axis.is_batch and not axis.is_recurrent)

    def recurrent_axis(self):
        """
        Returns:
            The tensor's recurrent Axis or None if there isn't one.
        """
        for axis in self:
            if axis.is_recurrent:
                return axis

    def flatten(self, force=False):
        """
        Produces flattened form of axes

        Args:
            force: Add a FlattenedAxis even when the axis is already flat. This is needed
             when the flatten is balanced by a later unflatten, as in dot.

        Returns:
            A flat axis.

        """
        if not force and len(self) == 1:
            return self[0]
        return FlattenedAxis(self)

    def set_shape(self, shape):
        """
        Set shape of Axes

        Args:
            shape: tuple or list of shapes, must be the same length as the axes
        """
        if len(shape) != len(self._axes):
            raise ValueError("shape's length %s must be equal to axes' length"
                             "%s" % (len(shape), len(self)))
        for axis, length in zip(self._axes, shape):
            axis.length = length

    def find_by_name(self, name):
        return Axes(axis for axis in self if axis.name == name)

    def __iter__(self):
        return self._axes.__iter__()

    def __len__(self):
        return len(self._axes)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Axes(self._axes.__getitem__(item))
        else:
            return self._axes.__getitem__(item)

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def __add__(self, other):
        """
        Returns list concatenated axes. Throws exception when there are Axis
        duplication.

        Arguments:
            other: the right-hand side operator axes

        Returns:
            current axes concatenated with the other axes
        """
        # self and other could not have common element
        other = make_axes(other)
        common_axes = self & other
        if len(common_axes) != 0:
            raise ValueError("Trying to concatenate %s with %s, but they have"
                             "common axes %s, which is not allowed."
                             % (self, other, common_axes))
        return make_axes(tuple(self) + tuple(other))

    def __sub__(self, other):
        """
        Returns ordered set difference of axes.

        Arguments:
            other: the right-hand side operator axes

        Returns:
            The ordered set difference of axes
        """
        other = make_axes(other)
        return make_axes((axis for axis in self if axis not in other))

    def __or__(self, other):
        """
        Returns ordered set union of axes.

        Arguments:
            other: the right-hand side operator axes

        Returns:
            The ordered set union of axes
        """
        other = make_axes(other)
        return make_axes(tuple(self) +
                         tuple(axis for axis in Axes(other) if axis not in self))

    def __and__(self, other):
        """
        Returns ordered set intersection of axes.

        Arguments:
            other: the right-hand side operator axes

        Returns:
            The ordered set intersection of axes
        """
        other = make_axes(other)
        return make_axes((axis for axis in self._axes if axis in other))

    def __eq__(self, other):
        """
        True if each ``Axis`` are matching and in same order (list comparison)

        Arguments:
            other: the right-hand side operator axes

        Returns:
            bool, True if each ``Axis`` are matching and in same order

        See Also ``is_equal_set`` if you want the comparison to ignore the Axes order
        """
        if not isinstance(other, Axes):
            raise ValueError((
                'other must be of type Axes, found type {}'
            ).format(type(other)))

        return self._axes.__eq__(other._axes)

    def __ne__(self, other):
        """
        The opposite of __eq__, True if not all ``Axis`` are matching or in
        different order (list comparison)

        Arguments:
            other: the right-hand side operator axes

        Returns:
            bool, True if not all ``Axis`` are matching or in different order
        """
        return not self == other

    def __nonzero__(self):
        """ Axes considered nonzero if axes are nonzero. """
        return bool(self._axes)

    def __hash__(self):
        return hash(self._axes)

    def is_sub_set(self, other):
        """
        Returns true if other is subset of self, i.e. <=

        Arguments:
            other: the right-hand side operator axes

        Returns:
            bool, true if other is subset of self
        """
        return set(self.names).issubset(set(make_axes(other).names))

    def is_super_set(self, other):
        """
        Returns true if other is superset of self, i.e. >=

        Arguments:
            other: the right-hand side operator axes

        Returns:
            bool, true if other is superset of self
        """
        return not self.is_sub_set(other)

    def is_equal_set(self, other):
        """
        Returns true if other has the same set of Axis names as self

        Arguments:
            other: the right-hand side operator axes

        Returns:
            bool, true if other has the same set of Axis names as self
        """
        return set(self.names) == set(make_axes(other).names)

    def is_not_equal_set(self, other):
        """
        Returns true if other does not the same set of Axis names as self

        Arguments:
           other: the right-hand side operator axes

        Returns:
           bool, true if other does not has the same set of Axis names as self
        """
        return not self.is_equal_set(other)

    @property
    def T(self):
        return Axes(axis.T for axis in self)

    def index(self, axis):
        """
        Returns the index of an axis

        Arguments:
            axis: The axis to search for.

        Returns:
            The index.
        """
        return self._axes.index(axis)

    @staticmethod
    @with_args_as_axes
    def assert_valid_broadcast(axes, new_axes):
        """
        Checks whether axes can be broadcasted to new_axes. We require
        that the components of axes be laid out in the same order in new_axes.

        Axes:
            axes: The original axes.
            new_axes: The broadcasted axes.

        Returns:
            True if axes can be broadcasted to new_axes, False otherwise.
        """
        removed_axes = axes - new_axes

        if removed_axes:
            raise ValueError(("The new_axes of a broadcast operation must "
                              "include all of the axes from the origional set "
                              "of axes. \n"
                              "  original axes: {axes}\n"
                              "  new axes: {new_axes}\n"
                              "  missing axes: {removed_axes}").format(
                axes=axes,
                new_axes=new_axes,
                removed_axes=removed_axes,
            ))

    @staticmethod
    @with_args_as_axes
    def is_valid_flatten_or_unflatten(src_axes, dst_axes):
        """
        Checks whether we can flatten OR unflatten from src_axes to dst_axes.

        The requirements are that the components of axes should all be
        present in new_axes and that they should be laid out in the same
        order. This check is symmetric.
        """

        # inflate
        src_axes = Axes.as_flattened_list(src_axes)
        dst_axes = Axes.as_flattened_list(dst_axes)

        # check equal number of Axis
        if len(src_axes) != len(dst_axes):
            return False

        # check all Axis are equal
        equal = [src == dst for src, dst in zip(src_axes, dst_axes)]
        return all(equal)

    @staticmethod
    @with_args_as_axes
    def assert_valid_flatten(unflattend_axes, flattened_axes):
        """
        Checks whther axes can safely be flattened to produce new_axes.
        The requirements are that the components of axes should all be
        present in new_axes and that they should be laid out in the same
        order.

        Arguments:
            unflattend_axes: The original axes.
            flattened_axes: The flattened axes.

        Returns:
            True if axes can be safely flattened to new_axes, False otherwise.
        """
        if not Axes.is_valid_flatten_or_unflatten(unflattend_axes, flattened_axes):
            raise ValueError("Trying to flatten:\n%s\nto:\n%s.\n"
                             "But they are of different lengths, or the axes"
                             "layouts are different"
                             % (unflattend_axes, flattened_axes))

    @staticmethod
    @with_args_as_axes
    def assert_valid_unflatten(flattened_axes, unflattend_axes):
        """
        Checks whether axes can safely be unflattened to produce new_axes.
        The requirements are that the components of axes should all be
        present in new_axes and that they should be laid out in the same
        order.

        Arguments:
            flattened_axes: The original axes.
            unflattend_axes: The unflattened axes.

        Returns:
            True if axes can be safely unflattened to new_axes, False otherwise.
        """
        if not Axes.is_valid_flatten_or_unflatten(flattened_axes, unflattend_axes):
            raise ValueError("Trying to unflatten:\n%s\nto:\n%s.\n"
                             "But they are of different lengths, or the axes"
                             "layouts are different"
                             % (unflattend_axes, flattened_axes))

    @property
    def size(self):
        """
        TODO: delete this method, the size should come from the tensor
        """
        return int(np.prod(self.lengths))

    def __repr__(self):
        return 'Axes({})'.format(
            ', '.join(map(repr, self))
        )

    def __str__(self):
        return ', '.join(map(str, self))

    @staticmethod
    def as_nested_list(axes):
        """
        Converts Axes to a list of axes with flattened axes expressed as nested lists

        Returns:
            Nested list of Axis objects
        """
        if isinstance(axes, (Axes, list)):
            return [Axes.as_nested_list(a) for a in axes]
        elif isinstance(axes, FlattenedAxis):
            return [Axes.as_nested_list(a) for a in axes.axes]
        elif isinstance(axes, Axis):
            return axes

    @staticmethod
    def as_flattened_list(axes):
        """
        Converts Axes to a list of axes with flattened axes expanded recursively.

        Returns:
            List of Axis objects
        """
        axes_list = [list(axis.axes) if axis.is_flattened else [axis]
                     for axis in axes]
        axes = list(itertools.chain.from_iterable(axes_list))

        # inflate recursively
        if any([axis.is_flattened for axis in axes]):
            return Axes.as_flattened_list(axes)
        else:
            return axes


class DuplicateAxisNames(ValueError):
    def __init__(self, message, duplicate_axis_names):
        super(DuplicateAxisNames, self).__init__(message)

        self.duplicate_axis_names = duplicate_axis_names


class IncompatibleAxesError(ValueError):
    pass


class UnmatchedAxesError(IncompatibleAxesError):
    pass


class AxesMap(frozendict):
    """
    AxesMap provides a way to define a axis name mapping: {Axis.name: Axis.name} and
    then apply this mapping to an Axes and get new Axes out.

    Right now AxesMap is implemented as immutible because I didn't want to deal with
    enforcing _assert_valid_axes_map on every method which mutates a dict and I didn't
    need a mutable datastructure anyway.  Feel free to make it mutable and add in
    invariant enforcement.
    """
    def __init__(self, *args, **kwargs):
        def replace_axis_with_name(x):
            if isinstance(x, Axis):
                return x.name
            return x

        # strip axis objects into just names
        super(AxesMap, self).__init__({
            replace_axis_with_name(k): replace_axis_with_name(v)
            for k, v in dict(*args, **kwargs).items()
        })

        self._assert_valid_axes_map()

    def map_axes(self, axes):
        """
        Returns:
            Axes with lengths from axes and names which have been passed through axes_map
        """
        return make_axes([self._map_axis(old_axis) for old_axis in axes])

    def _map_axis(self, old_axis):
        """
        Given a map from {old_axes_name: new_axes_name} and an old_axis map the
        old_axis into the new_axes.
        """
        if old_axis.name in self:
            return make_axis(old_axis.length, self[old_axis.name])
        else:
            return old_axis

    def _duplicate_axis_names(self):
        """
        Returns:
            a dictionary mapping to duplicate target names and the source names
            that map to it: {target: set([source, ...])}
        """
        # invert axes_map to see if there are any target axis names that are
        # duplicated
        counts = collections.defaultdict(set)
        for key, value in self.items():
            counts[value].add(key)

        # filter counts to include only duplicate axis
        return {x: y for x, y in counts.items() if len(y) > 1}

    def _assert_valid_axes_map(self):
        """
        Ensure that there are no axis which map to the same axis and raise a
        helpful error message.
        """
        duplicate_axis_names = self._duplicate_axis_names()

        # if there are duplicate_axis_names throw an exception
        if duplicate_axis_names:
            message = 'AxesMap was can not have duplicate names, but found:'
            for target_axis, source_axes in duplicate_axis_names.items():
                message += '\n    {} maps to {}'.format(
                    target_axis, ', '.join(source_axes)
                )

            raise DuplicateAxisNames(message, duplicate_axis_names)

    def invert(self):
        return {v: k for k, v in self.items()}


def _reduce_nested(elem, agg, func):
    """
    Reduces a nested sequence by applying a function to each
    of its elements and returns an aggregation.

    Arguments:
      elem: The object to be reduced, either a sequence
        or a singleton.
      agg: A variable holding information collected
        as the sequence is collapsed.
      func: A function to augment the aggregate by processing
        a singleton. Should have the form func(agg, elem) -> agg

    Returns:
        agg: The final aggregate returned by the function.
    """
    if isinstance(elem, collections.Iterable):
        for sub in elem:
            agg = _reduce_nested(sub, agg, func)
        return agg
    else:
        return func(agg, elem)


class FlattenedAxis(Axis):
    """
    A FlattenedAxis has length which is the product of the lengths of all
    Axis in the axes.  The original Axes object is stored so that we can later
    unflatten this Axis back to its original component Axis.

    Notes: since we allows Axis to have duplicated names globally, NameableValue
    is not used here.
    """

    __name_counter = 0

    def __init__(self, axes, **kwargs):
        # get length
        axes = Axes(axes)
        if len(axes) == 1 and axes[0].is_flattened:
            pass
        length = reduce(operator.mul, axes.lengths, 1)

        # set name
        name = '%s_%s' % (type(self).__name__, type(self).__name_counter)
        type(self).__name_counter += 1

        # parent constructor
        super(FlattenedAxis, self).__init__(length=length, name=name, **kwargs)
        self._axes = axes

    @property
    def is_flattened(self):
        """
        Returns:
            True is this is a FlattendAxis.
        """
        return True

    @property
    def empty(self):
        """
        Returns:
            Whether this axes contains no collapsed axes.
        """
        return len(self._axes) == 0

    @property
    def single(self):
        """
        Returns:
            Whether this axes contains exactly one collapsed axes.
        """
        return len(self._axes) == 1

    @property
    def axes(self):
        """
        Returns:
            The flattened axes contained in this object.
        """
        return self._axes

    def __eq__(self, other):
        return other.is_flattened\
            and all(l == r for l, r in zip(self.axes, other.axes))

    def __hash__(self):
        return hash(self.axes)

    def __repr__(self):
        return 'FlattenedAxis(%s)' % ', '.join(repr(axis) for axis in self.axes)


def reduce_strides(strides):
    """
    Reduces a nested tuple describing the strides of a tensor
    into a tuple giving the stride of each of its dimensions.

    Arguments:
        strides: The nested tuple.

    Returns:
        strides: The tuple of strides.
    """
    return tuple(int(_reduce_nested(elem, float('inf'), min))
                 for elem in strides)


def _make_stride(inner_size, axis, fsz):
    """
    Generates a nested tuple that provides the striding information
    for an occurrence of axis. If the axis is a FlattenedAxis, the
    stride will be a tuple containing the strides of each collapsed
    axis. Otherwise, the stride will be an integer.

    Arguments:
        inner_size: The total size of all dimensions smaller than this
        axis, i.e. all axes to the right of this one when they are
        laid out in c-contiguous order.
        axis: The axis for which we are generating a stride.
        fsz: A nested tuple supplying the sizes of each dimension collapsed
        into the axis. The size may be larger than the length of the axis.

    Returns:
        inner_size: The total size of this axis and all smaller dimensions.
        stride: The stride given to the axis.
    """
    if axis.is_flattened:
        return _make_strides(inner_size, axis.axes, fsz)
    else:
        stride = inner_size
        inner_size *= fsz
        return inner_size, stride


def _make_strides(inner_size, axes, full_sizes):
    """
    Generates a tuple of strides for a set of axes. See _make_stride
    for a description of the stride given to each axis.

    Arguments:
        inner_size: The total size of all dimensions smaller than
        the axes.
        axes: The axes for which we are generating strides.
        full_sizes: The size of each axis.

    Returns:
        inner_size: The total size of these axes and all smaller dimensions.
        strides: The strides generated for the axes.
    """
    full_strides = []
    for axis, fsz in reversed(list(zip(axes, full_sizes))):
        inner_size, stride = _make_stride(inner_size, axis, fsz)
        full_strides.append(stride)
    return inner_size, tuple(reversed(full_strides))


class TensorDescription(NameableValue):
    """
    Description of a tensor that will be allocated in hardware.

    Names the tensor's dimensions with axes and holds pointers to the
    buffer allocated by the analysis and the backend tensor value
    (e.g. a cpu or gpu tensor).

    Arguments:
        axes: Axes of the tensor.
        base: If a view, the viewed tensor's description.
        dtype: The type of the tensor.
        full_strides: The strides of each axis.
        full_sizes: The allocated size of each axis (may be larger than the axis).
        offset: An offset into the viewed tensor.
        next_tensor_decription: In a reshape, tensor description of reshaped tensor.
        is_persistent: The tensor should be persistent, i.e. survive from computation to
            computation.
        is_input: The device tensor can be written from the host.
        **kwargs: Additional args for related classes.
    """

    def __init__(self, axes, base=None,
                 layout=None,
                 dtype=None,
                 full_strides=None, full_sizes=None, offset=0,
                 next_tensor_description=None,
                 is_persistent=False,
                 is_input=False,
                 is_placeholder=False,
                 op=None,
                 **kwargs):
        super(TensorDescription, self).__init__(**kwargs)
        # TODO: get the default type from the backend. May not always be numpy.
        # TODO: support flattening, unflattening, other complex reshapes
        axes = Axes(axes)
        self.axes = axes
        self.__layout = layout
        self.__value = None
        self.__buffer = None
        self.__register = None
        self.__base = base
        self.dtype = default_dtype(dtype)
        self.offset = offset
        self.ndim = len(self.axes)
        self.full_sizes = tuple(full_sizes) if full_sizes is not None \
            else self.axes.full_lengths
        self.next_tensor_description = next_tensor_description
        self.__is_persistent = is_persistent
        self.__is_input = is_input
        self.__is_placeholder = is_placeholder
        self.op = op
        if not isinstance(self.name, str):
            raise ValueError()
        for axis in axes:
            if axis.length is None:
                raise ValueError((
                    'axes used in the constructor of TensorDescription must '
                    'always have non-None length.  Axis {axis} has length '
                    'None.'
                ).format(axis=axis))

        if full_strides is None:
            _, full_strides = _make_strides(
                self.dtype.itemsize,
                self.axes,
                self.full_sizes
            )
            self.full_strides = full_strides
        else:
            self.full_strides = tuple(full_strides)

        assert len(self.full_sizes) == self.ndim, \
            "Sizes must have same number of dimensions as axes"
        assert len(self.full_strides) == self.ndim, \
            "Strides must have same number of dimensions as axes"


    def __repr__(self):
        return self.base.name

    @property
    def is_persistent(self):
        """

        Returns: True if persists from computation to computation.

        """
        if self.base is self:
            return self.__is_persistent
        return self.base.is_persistent

    @property
    def is_input(self):
        """

        Returns: True if writable from host.

        """
        if self.base is self:
            return self.__is_input
        return self.base.is_input

    @property
    def is_placeholder(self):
        """

        Returns: True if a placeholder; a place to attach a tensor.

        """
        if self.base is self:
            return self.__is_placeholder
        return self.base.is_placeholder

    @property
    def parameter_key(self):
        """
        Returns: A tuple that can be used to tell if two views of a tensor are equivalent.
        """
        return (self.shape, self.dtype, self.offset, self.strides, self.layout)

    @property
    def axes_key(self):
        return (self.axes, self.shape, self.dtype, self.offset, self.strides, self.layout)

    def flatten(self, new_axes):
        """
        Flattens a tensor description to give it the Axes in new_axes.
        See Axes.assert_valid_flatten for a description of permitted values of new_axes.

        Arguments:
            new_axes: The Axes of the flattened tensor description.

        Returns:
            The reshaped tensor description.
        """
        new_axes = Axes(new_axes)
        Axes.assert_valid_flatten(self.axes, new_axes)

        new_strides = []
        new_sizes = []
        idx = 0
        for new_axis in new_axes:
            if new_axis == self.axes[idx]:
                new_stride = self.full_strides[idx]
                new_size = self.full_sizes[idx]
                idx += 1
            else:
                l = len(new_axis.axes)
                new_stride = self.full_strides[idx:idx + l]
                new_size = self.full_sizes[idx:idx + l]
                idx += l

            new_strides.append(new_stride)
            new_sizes.append(new_size)

        return TensorDescription(
            new_axes,
            base=self.base,
            dtype=self.dtype,
            full_strides=new_strides,
            full_sizes=new_sizes,
            offset=self.offset,
            next_tensor_description=self,
            name = self.name + 'rFlatten'
        )

    def unflatten(self, new_axes):
        """
        Unflattens a tensor description to give it the Axes in new_axes.
        See Axes.assert_valid_unflatten for a description of the permitted values of
        new_axes

        Arguments:
            new_axes: The Axes of the unflattened TensorDescription.

        Returns:
            The unflattened tensor description.
        """
        def find_axis_stride_and_length(axis):
            """
            Find the stride and length for an axis.

            Start at the current tensor description and then work back
            through reshapings of it looking for a mention of the axis
            that can be used to determine the storage stride and offset.

            Args:
                axis: The axis.

            Returns:
                stride, length of axis

            """
            td = self
            while td is not None:
                for idx, a in enumerate(td.axes):
                    # Try to find a match for axis in this td
                    full_strides = td.full_strides[idx]
                    full_sizes = td.full_sizes[idx]
                    if a == axis:
                        return full_strides, full_sizes

                    if a.is_flattened:
                        # Can be embedded ina a flattened axis description
                        if not isinstance(full_strides, tuple):
                            # An axis cast can lose striding info, so need to
                            # recreate it from the axis lengths. Being flattened
                            # implies C-contiguous
                            stride = full_strides
                            full_strides = []
                            full_sizes = []
                            for s in reversed(a.axes):
                                full_sizes.insert(0, s.length)
                                full_strides.insert(0, stride)
                                stride = stride * s.length

                        # Now search for axis in the flattened axis
                        for sub_idx, b in enumerate(a.axes):
                            if b == axis:
                                return full_strides[sub_idx], full_sizes[sub_idx]

                # Move on to the next tensor description in the reshaping chain
                td = td.next_tensor_description

            # Sometimes we just don't have enough information.
            raise ValueError()

        new_axes = Axes(new_axes)
        Axes.assert_valid_unflatten(self.axes, new_axes)

        new_strides = []
        new_sizes = []
        for new_axis in new_axes:
            stride, size = find_axis_stride_and_length(new_axis)
            new_strides.append(stride)
            new_sizes.append(size)

        return TensorDescription(
            new_axes,
            base=self.base,
            dtype=self.dtype,
            full_strides=new_strides,
            full_sizes=new_sizes,
            offset=self.offset,
            next_tensor_description=self,
            name=self.name + 'rUnflatten',
        )

    def transpose(self):
        """
        Reverses the axes of the tensor description.

        Retuns:
            A tensor description with the axes reversed.
        """
        new_axes = reversed(self.axes)
        full_sizes = reversed(self.full_sizes)
        full_strides = reversed(self.full_strides)
        return TensorDescription(
            Axes(new_axes),
            base=self.base,
            dtype=self.dtype,
            full_strides=tuple(full_strides),
            full_sizes=tuple(full_sizes),
            offset=self.offset,
            next_tensor_description=self,
            name=self.name + 'rTranspose',
        )

    def clone(self):
        """
        Creates a copy of this tensor description

        Retuns:
            A copy of this tensor description
        """
        return TensorDescription(
            self.axes,
            base=self.base,
            dtype=self.dtype,
            full_strides=self.full_strides,
            full_sizes=self.full_sizes,
            offset=self.offset,
            next_tensor_description=self.next_tensor_description,
            name=self.name + 'cView',
        )

    def broadcast(self, new_axes):
        """
        Adds axes to a tensor description to give it a new shape.
        See Axes.assert_valid_broadcast for a description of the permitted
        transformations.

        Arguments:
            new_axes: The axes of the broadcasted tensor description.

        Returns:
            TensorDescription: The broadcasted tensor description.
        """
        Axes.assert_valid_broadcast(self.axes, new_axes)
        return self.reorder_and_broadcast(new_axes)

    def reorder(self, new_axes):
        """
        Shuffles axes of a tensor to give it a new shape. The axes of
        this tensor description and new_axes must have the same elements.

        Arguments:
            new_axes: The axes of the reordered tensor.

        Returns:
            TensorDescription: The reordered tensor description.
        """
        if not self.axes.is_equal_set(new_axes):
            raise ValueError((
                "Reorder can't change which axes are available, only the "
                "order.  {} and {} are different sets, not just order."
            ).format(self, new_axes))

        return self.reorder_and_broadcast(new_axes)

    def reorder_and_broadcast(self, new_axes):
        """
        Adds or shuffles axes to give a tensor description a new shape.
        This function is used to implement broadcast and reorder.

        Arguments:
            new_axes: The axes of the broadcasted or reordered tensor.

        Returns:
            TensorDescription: A description of the tensor after the
            transformation.
        """
        def zero_in_shape(tup):
            if isinstance(tup, collections.Iterable):
                return tuple(
                    zero_in_shape(t) for t in tup
                )
            else:
                return 0

        new_axes = Axes(new_axes)
        new_strides = []
        new_sizes = []
        for axis in new_axes:
            if axis in self.axes:
                idx = self.axes.index(axis)
                new_strides.append(self.full_strides[idx])
                new_sizes.append(self.full_sizes[idx])
            elif axis.is_flattened:
                lengths = axis.axes.full_lengths
                new_strides.append(zero_in_shape(lengths))
                new_sizes.append(lengths)
            else:
                new_strides.append(0)
                new_sizes.append(axis.length)

        return TensorDescription(
            new_axes,
            base=self.base,
            dtype=self.dtype,
            full_strides=new_strides,
            full_sizes=new_sizes,
            offset=self.offset,
            next_tensor_description=self,
            name=self.name + 'rReorderBroadcast',
        )

    def cast(self, new_axes):
        """
        Return a tensor desciption for a view of the tensor.

        Arguments:
            new_axes: The axes for the view.

        Returns:
            The tensor description.

        """
        full_strides = self.full_strides
        full_sizes = self.full_sizes
        if self.ndim == 0:
            full_strides = (0,) * len(new_axes)
            full_sizes = new_axes.full_lengths

        return TensorDescription(
            new_axes,
            base=self.base,
            dtype=self.dtype,
            full_strides=full_strides,
            full_sizes=full_sizes,
            offset=self.offset,
            next_tensor_description=self,
            name=self.name + 'rCast',
        )

    def slice(self, slices, new_axes):
        """
        Return a tensor description for a slice view of this tensor.

        Arguments:
            slices: The slices to take from the tensor, each of which is
            either an integer or a python slice. If the input has too few
            axes for the tensor, we assume that the entire axis should be
            taken for dimensions towards the end of the tensor.
            new_axes: the axes to use as labels for the sliced tensor.

        Returns:
            The tensor description for the slice.
        """
        slices = list(slices)
        while len(slices) < self.ndim:
            slices.append(slice(None))

        offset = self.offset
        full_strides = []
        full_sizes = []
        new_index = 0

        # check new_axes for the correct length
        num_dimensions_out = len([s for s in slices if isinstance(s, slice)])
        if len(new_axes) != num_dimensions_out:
            raise ValueError((
                'in a slice operation, the number of axes passed in to '
                'new_axes ({num_new_axes}) must be the same as the number of '
                'slice objects in slices ({num_slices}).'
            ).format(
                num_new_axes=len(new_axes),
                num_slices=num_dimensions_out,
            ))

        for s, axis, stride, size in zip(slices, self.axes, self.strides, self.sizes):
            if isinstance(s, slice):
                # only increment new_axis when the input slice is a slice and
                # not a integer
                new_axis = new_axes[new_index]
                new_index += 1

                # ensure slice is of the kind we support
                _validate_slice(s)

                # ensure new_axis has the correct length
                new_axis.length = _sliced_length(s, axis.length)

                start, stop, step = s.indices(axis.length)

                full_strides.append(stride * step)
                full_sizes.append(size)

                idx = start
            else:
                # this is a simple integer slice, ex: y = x[1]
                idx = s

            # TODO: write a test that fails if abs() is removed
            offset += idx * abs(stride)

        return TensorDescription(
            new_axes,
            base=self.base,
            dtype=self.dtype,
            full_strides=tuple(full_strides),
            full_sizes=tuple(full_sizes),
            offset=offset,
            next_tensor_description=self,
            name=self.name + "rSlice",
        )

    @property
    def shape(self):
        """
        Returns: The shape of the tensor.
        """
        return self.axes.lengths

    @property
    def strides(self):
        """The strides of the tensor."""
        return reduce_strides(self.full_strides)

    @property
    def sizes(self):
        """The allocated sizes for each axis."""
        return tuple(_reduce_nested(_, 1, operator.mul)
                     for _ in self.full_sizes)

    @property
    def tensor_size(self):
        result = self.dtype.itemsize
        for s in self.sizes:
            result = result * s
        return result

    @property
    def c_contiguous(self):
        """

        Returns:
            True if the tensor's strides are row-major contiguous.
        """
        s = self.dtype.itemsize
        cstrides = []
        for _ in reversed(self.shape):
            cstrides.insert(0, s)
            s = s * _
        return tuple(cstrides) == self.strides

    @property
    def broadcast_contiguous(self):
        """
        Returns:
            True if tensor's strides are contiguous or broadcasted
        """
        if self.shape == ():
            return True

        broadcast_axes = np.where(np.equal(self.strides, 0))[0]
        aug_shape = list(self.shape)
        for bcast_axis in broadcast_axes:
            aug_shape[bcast_axis] = 1

        s = self.dtype.itemsize
        cstrides = []
        for _ in reversed(aug_shape):
            cstrides.insert(0, s)
            s = s * _

        for bcast_axis in broadcast_axes:
            cstrides[bcast_axis] = 0
        return tuple(cstrides) == self.strides

    @property
    def base(self):
        """The viewed tensor description or None if not a view."""
        return self.__base or self

    @property
    def layout(self):
        """The layout of the underlying storage."""
        return self.__layout

    @layout.setter
    def layout(self, value):
        """
        Sets the backend-specific memory layout to be used by the tensor.

        Arguments:
          value: the layout to use

        Returns:
        """
        self.__layout = value

    @property
    def register(self):
        return self.base.__register

    @register.setter
    def register(self, value):
        self.base.__register = value

    def is_base(self):
        """This tensor provides its own storage."""
        return self.__base is None
