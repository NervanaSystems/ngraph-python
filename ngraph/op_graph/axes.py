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

import collections
import operator
from functools import reduce, wraps

import numpy as np
import types
from weakref import WeakValueDictionary
from abc import ABCMeta
from builtins import object, map, range, zip
from future.utils import with_metaclass

from ngraph.util.names import NameableValue


def default_dtype(dtype=None):
    if dtype is None:
        dtype = np.dtype(np.float32)
    return dtype


def default_int_dtype(dtype=None):
    if dtype is None:
        dtype = np.dtype(np.int32)
    return dtype


class Axis(with_metaclass(ABCMeta, NameableValue)):
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
        match_on_length: Whether to only use length (and not identity) when comparing
            equality against other Axis values. This is useful for anonymous Axis of
            Constant tensors.
    """
    def __init__(self,
                 length=None,
                 batch=False,
                 recurrent=False,
                 match_on_length=False,
                 **kwargs):
        super(Axis, self).__init__(**kwargs)
        self.__length = length
        self.batch = batch
        self.recurrent = recurrent
        self.match_on_length = match_on_length
        self.duals = WeakValueDictionary()

    @property
    def batch(self):
        """
        Whether the axis is a batch axis.
        """
        return self.__batch

    @batch.setter
    def batch(self, value):
        self.__batch = value

    @property
    def recurrent(self):
        """
        Whether the axis is a recurrent axis.
        """
        return self.__recurrent

    @recurrent.setter
    def recurrent(self, value):
        self.__recurrent = value

    @property
    def length(self):
        """
        Returns:
            The length of the axis.
        """
        return self.__length

    @length.setter
    def length(self, value):
        self.__length = value

    @property
    def axes(self):
        return Axes([self])

    @property
    def dual_level(self):
        """

        Returns:
            Axis displacement for dot.

            In dot, left axis of level n matches right axis of level n+1. Level n-1 is
            the dual space of level n.

        """
        return 0

    def get_dual(self, offset=-1):
        """
        Get the dual of this axis.

        Args:
            offset: How many duals, default is -1.

        Returns:
            A dual axis.

        """
        if offset == 0:
            return self
        dual = self.duals.get(offset, None)
        if dual is None:
            dual = DualAxis(self, offset)
            self.duals[offset] = dual
        return dual

    def get_transpose(self):
        return self.primary_axis.get_dual(-1 - self.dual_level)

    @property
    def primary_axis(self):
        return self

    def __repr__(self):
        return 'Axis({name}: {length})'.format(name=self.name, length=self.length)

    def __eq__(self, other):
        if not isinstance(other, Axis):
            return False
        elif self.match_on_length or other.match_on_length:
            return self.length == other.length
        return self is other

    def __hash__(self):
        return id(self)


class DualAxis(Axis):
    """
    A DualAxis is returned from Axis.get_dual. It shares length with the primary axis.

    This class should only be constructed by Axis.get_dual.
    """
    def __init__(self, primary_axis, dual_level):
        super(DualAxis, self).__init__()
        self.__primary_axis = primary_axis
        self.__dual_level = dual_level

    @property
    def length(self):
        return self.__primary_axis.length

    @property
    def primary_axis(self):
        return self.__primary_axis

    @property
    def dual_level(self):
        return self.__dual_level

    def get_dual(self, offset=-1):
        """
        Get the dual of this axis.

        Args:
            offset: How many duals, default is -1.

        Returns:
            A dual axis.

        """
        return self.primary_axis.get_dual(self.dual_level + offset)

    def __repr__(self):
        return 'DualAxis({axis}:{level})'.format(axis=self.primary_axis, level=self.dual_level)


class FunctionAxis(Axis):
    """
    A function axis is an axis whose length is computed by a user-supplied function.

    Instances should only be created internally because using a
    function that changes the length after a transformation will result in
    undefined behaviour.

    Currently, this class is only used by the SlicedAxis and PaddedAxis subclasses,
    which derive their length from a parent axis's length. This satisfies the above
    restriction, because we expect the parent axis to become immutable once
    the transformation begins.
    """
    def __init__(self, parent, length_fun, **kwargs):
        super(FunctionAxis, self).__init__(length=-1, **kwargs)
        self.length_fun = length_fun
        self.batch = parent.batch
        # TODO: self.recurrent = parent.recurrent

    @property
    def length(self):
        return self.length_fun()


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


class SlicedAxis(FunctionAxis):
    """
    An axis created by slicing a parent axis.

    The length is computed dynamically from the length of the parent.

    Arguments:
        parent: The axis being sliced.
        s: The slice.
        kwargs: Arguments for related classes.

    TODO: Right now, a 0 length slice is allowed.  Perhaps we want to raise an
    exception instead?
    """
    def __init__(self, parent, s, **kwargs):
        self.parent = parent
        self.slice = s

        _validate_slice(s)

        super(SlicedAxis, self).__init__(
            parent=parent,
            length_fun=lambda: _sliced_length(s, parent.length),
            **kwargs
        )

    def __repr__(self):
        return (
            'SlicedAxis({name}: {length}; parent: {parent}; slice: {slice})'
        ).format(
            name=self.name,
            length=self.length,
            parent=self.parent,
            slice=self.slice,
        )


class PaddedAxis(FunctionAxis):
    """
    An axis created by padding a parent axis.

    Arguments:
        parent: The axis being padded.
        pad: A two-element array of pre and post padding.
    """
    def __init__(self, parent, pad, **kwargs):
        self.parent = parent
        self.pad = pad

        def padded_length():
            return parent.length + pad[0] + pad[1]

        super(PaddedAxis, self).__init__(
            parent=parent, length_fun=padded_length, **kwargs
        )

    def __repr__(self):
        return (
            'PaddedAxis({name}: {length}; parent: {parent}; pad: {pad})'
        ).format(
            name=self.name,
            length=self.length,
            parent=self.parent,
            pad=self.pad,
        )


def no_duplicates(arr):
    """
    Returns whether there are duplicates in a list. The elements in the array
    should be hashable.

    Arguments:
        arr: The list to check.

    Returns:
        bool: True if there are no duplicates, False if there are.
    """
    s = set()
    for x in arr:
        if x in s:
            return False
        s.add(x)
    return True


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

        if not no_duplicates(axes):
            raise ValueError(
                'The axes labels of a tensor cannot contain duplicates.'
            )
        self._axes = tuple(axes)

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
        return tuple(x.axes.full_lengths if isinstance(x, FlattenedAxis)
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
            The Axes subset that are batch axes.
        """
        return Axes(axis for axis in self if axis.batch)

    def sample_axes(self):
        """
        Returns:
            The Axes subset that are not batch axes.
        """
        return Axes(axis for axis in self if not axis.batch)

    def recurrent_axes(self):
        """
        Returns:
            The Axes subset that are recurrent axes.
        """
        return Axes(axis for axis in self if axis.recurrent)

    def flatten(self):
        if len(self) == 1:
            return self[0]
        return FlattenedAxis(self)

    def shape_dict(self):
        """
        Retuns:
            dict: A dictionary with names of the axes as keys and
            lengths as values
        """
        names = [axis.name for axis in self._axes]
        # TODO: get rid of this temporary hack.
        names = [name.split('_')[0] for name in names]
        short_names = []
        for name in names:
            if name.find('.') != -1:
                name = name.split('.')[1]
            short_names.append(name)
        vals = [axis.length for axis in self._axes]
        return dict(zip(short_names, vals))

    def set_shape(self, shape):
        axes = self._axes
        diff = len(axes) - len(shape)
        if diff == 0:
            for axis, length in zip(axes, shape):
                axis.length = length
            return

        if diff > 0:
            axes[0].length = shape[0]
            for i in range(1, diff + 1):
                # Pad missing dimensions with 1.
                axes[i].length = 1
            for length in shape[diff:]:
                i += 1
                axes[i].length = length
            return
        raise ValueError('Number of axes %d too low for shape %s' % (
                         len(axes), shape))

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
        return Axes.concatenate(self, Axes(other))

    def __sub__(self, other):
        return Axes.subtract(self, Axes(other))

    def __eq__(self, other):
        if not isinstance(other, Axes):
            raise ValueError((
                'other must be of type Axes, found type {}'
            ).format(type(other)))

        return self._axes.__eq__(other._axes)

    def __ne__(self, other):
        return not self == other

    def __nonzero__(self):
        """ Axes considered nonzero if axes are nonzero. """
        return bool(self._axes)

    def __hash__(self):
        return hash(self._axes)

    def get_dual(self, dual_offset=-1):
        return Axes((axis.get_dual(dual_offset) for axis in self))

    def get_transpose(self):
        return Axes(axis.get_transpose() for axis in self)

    @staticmethod
    @with_args_as_axes
    def concatenate(axes1, axes2):
        """
        Returns the union of the elements, leaving out duplicate Axes.

        Arguments:
            axes1: first axes to concatenate
            axes2: second axes to concatenate

        Returns:
            The ordered union
        """
        assert isinstance(axes1, Axes) and isinstance(axes2, Axes)
        return Axes(
            tuple(axes1) +
            tuple(axis for axis in axes2 if axis not in axes1)
        )

    @staticmethod
    @with_args_as_axes
    def subtract(axes1, axes2):
        """
        Returns the difference of the two Axes.

        Arguments:
            axes1: first axes to subtract
            axes2: second axes to subtract

        Returns:
            The ordered difference
        """
        assert isinstance(axes1, Axes) and isinstance(axes2, Axes)
        return Axes((axis for axis in axes1 if axis.get_dual(0) not in axes2))

    @staticmethod
    @with_args_as_axes
    def intersect(axes1, axes2):
        """
        Returns the intersection of the elements, leaving out duplicate Axes.

        Arguments:
            axes1: first axes to intersect
            axes2: second axes to intersect

        Returns:
            The ordered intersection
        """
        assert isinstance(axes1, Axes) and isinstance(axes2, Axes)
        return Axes((axis for axis in axes1 if axis in axes2))

    @staticmethod
    @with_args_as_axes
    def linear_map_axes(in_axes, out_axes):
        """
        For tensors ``out = dot(T, in)`` used in linear transformations
        determines the axes ``T`` must have.

        Arguments:
            in_axes: The axes of ``in``.
            out_axes: The axes of ``out``.

        Returns:
            Axes of the weights used in the transformation.
        """
        return (
            out_axes + in_axes.get_dual()
        )

    @staticmethod
    @with_args_as_axes
    def find(axes, sub_axes):
        """
        Attempts to locate a subsequence of Axes (sub_axes) in axes.

        Arguments:
            axes: The superset of Axes.
            sub_axes: Axes to search for.

        Returns:
            int: The index at which the subsequence sub_axes occurs in
            axes.
        """
        assert isinstance(axes, Axes) and isinstance(sub_axes, Axes)
        for i in range(len(axes) - len(sub_axes) + 1):
            if axes[i:i + len(sub_axes)] == sub_axes:
                return i
        raise ValueError('Could not find sub_axes')

    @staticmethod
    def find_axis(axes, axis):
        """
        Attempts to locate an axis in Axes.

        Arguments:
            axes: The superset of Axes.
            axis: Axis to search for.

        Returns:
            int: The index at which the axis occurs in axes.
        """
        axes = Axes(axes)
        assert isinstance(axis, Axis)
        return axes._axes.index(axis)

    @staticmethod
    def index(axes, axis):
        """
        Returns the index of an axis in Axes.

        Arguments:
            axes: The axes in which to search.
            axis: The axis to search for.

        Returns:
            The index.
        """
        axes = Axes(axes)
        return axes._axes.index(axis)

    @staticmethod
    @with_args_as_axes
    def same_elems(axes1, axes2):
        """
        Checks whether two Axes have the same elements.

        Arguments:
            axes1: First axes.
            axes2: Second axes.

        Returns:
            True if axes1 and axes2 have the same elements,
            False otherwise.
        """
        return set(axes1) == set(axes2)

    @staticmethod
    @with_args_as_axes
    def check_broadcast(axes, new_axes):
        """
        Checks whether axes can be broadcasted to new_axes. We require
        that the components of axes be laid out in the same order in new_axes.

        Axes:
            axes: The original axes.
            new_axes: The broadcasted axes.

        Returns:
            True if axes can be broadcasted to new_axes, False otherwise.
        """
        def check(condition):
            if not condition:
                return False

        axes_s = set(axes)
        idx = 0
        for new_axis in new_axes:
            if idx < len(axes) and new_axis == axes[idx]:
                idx += 1
            else:
                check(new_axis not in axes_s)

        check(idx == len(axes))
        return True

    @staticmethod
    @with_args_as_axes
    def check_flatten(axes, new_axes):
        """
        Checks whther axes can safely be flattened to produce new_axes.
        The requirements are that the components of axes should all be
        present in new_axes and that they should be laid out in the same
        order.

        Arguments:
            axes: The original axes.
            new_axes: The flattened axes.

        Returns:
            True if axes can be safely flattened to new_axes, False otherwise.
        """
        return Axes.check_unflatten(new_axes, axes)

    @staticmethod
    @with_args_as_axes
    def check_unflatten(axes, new_axes):
        """
        Checks whether axes can safely be unflattened to produce new_axes.
        The requirements are that the components of axes should all be
        present in new_axes and that they should be laid out in the same
        order.

        Arguments:
            axes: The original axes.
            new_axes: The unflattened axes.

        Returns:
            True if axes can be safely unflattened to new_axes, False otherwise.
        """
        if len(axes) == 0:
            return True

        def check(condition):
            if not condition:
                return False

        idx = 0
        for axis in axes:
            if axis == new_axes[idx]:
                idx += 1
            else:
                check(isinstance(axis, FlattenedAxis))
                new_idx = idx + len(axis.axes)
                check(axis.axes == new_axes[idx:new_idx])
                idx = new_idx
        check(idx == len(new_axes))
        return True

    # TODO: delete this method, the size should come from the tensor
    @property
    def size(self):
        """TODO."""
        size = 1
        for x in self:
            size *= x.length
        return size

    def __repr__(self):
        return 'Axes({})'.format(
            ', '.join(map(repr, self))
        )

    def append(self, axis):
        """
        Appends an axis

        Arguments:
            other: The Axis object to append.
        """
        self._axes = Axes(tuple(self) + (axis,))

    def insert(self, index, axis):
        """
        Inserts an axis
        Arguments:
            index   : Index to insert at
            axis    : The Axis object to insert
        """
        axes = self._axes
        axes.insert(index, axis)
        self._axes = Axes(axes)


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
    """

    def __init__(self, axes, **kwargs):
        axes = Axes(axes)
        if len(axes) == 1 and isinstance(axes[0], FlattenedAxis):
            pass
        length = reduce(operator.mul, axes.lengths, 1)
        super(FlattenedAxis, self).__init__(length=length, **kwargs)
        self.__axes = axes

    @property
    def empty(self):
        """
        Returns:
            Whether this axes contains no collapsed axes.
        """
        return len(self.__axes) == 0

    @property
    def single(self):
        """
        Returns:
            Whether this axes contains exactly one collapsed axes.
        """
        return len(self.__axes) == 1

    @property
    def axes(self):
        """
        Returns:
            The flattened axes contained in this object.
        """
        return self.__axes

    def __eq__(self, other):
        return isinstance(other, FlattenedAxis)\
            and all(l == r for l, r in zip(self.axes, other.axes))

    def __hash__(self):
        return hash(self.axes)

    def __repr__(self):
        s = 'FlattenedAxis('
        for i, x in enumerate(self.axes):
            s += repr(x)
            s += ', '
        s += ')'
        return s


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


def _check_sliced_axis_length(s, axis, new_axis):
    """
    Ensure that the length of the axis resulting from slicing axis with
    slice s matches the length of new_axis
    """

    expected_length = _sliced_length(s, axis.length)
    if expected_length != new_axis.length:
        raise ValueError((
            "A slice operation ({slice}) was attempted on axis "
            "{axis} with length {axis_length}.  The result of "
            "which is a new sliced axis of length "
            "{expected_length}.  The new_axis passed in "
            "{new_axis} has a different length which does not "
            "match: {new_axis_length}."
        ).format(
            slice=s,
            axis=axis,
            axis_length=axis.length,
            expected_length=expected_length,
            new_axis=new_axis,
            new_axis_length=new_axis.length,
        ))


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
    if isinstance(axis, FlattenedAxis):
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
    (e.g. a numpy or gpu tensor).

    Arguments:
        axes: Axes of the tensor.
        base: If a view, the viewed tensor's description.
        dtype: The type of the tensor.
        full_strides: The strides of each axis.
        full_sizes: The allocated size of each axis (may be larger than the axis).
        offset: An offset into the viewed tensor.
        **kwargs: Additional args for related classes.

    """

    def __init__(self, axes, base=None,
                 dtype=None,
                 full_strides=None, full_sizes=None, offset=0,
                 **kwargs):
        super(TensorDescription, self).__init__(**kwargs)
        # TODO: get the default type from the backend. May not always be numpy.
        # TODO: support flattening, unflattening, other complex reshapes
        axes = Axes(axes)
        self.axes = axes
        self.transformer = None
        self.__value = None
        self.__buffer = None
        self.__register = None
        self.__base = base
        self.dtype = default_dtype(dtype)
        self.offset = offset
        self.ndim = len(self.axes)
        self.__read_only = False
        self.full_sizes = tuple(full_sizes) if full_sizes is not None \
            else self.axes.full_lengths
        self.style = {}

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

    @property
    def parameter_key(self):
        """
        Returns: A tuple that can be used to tell if two views of a tensor are equivalent.
        """
        return (self.shape, self.dtype, self.offset, self.strides)

    def flatten(self, new_axes, name=None):
        """
        Flattens a tensor description to give it the Axes in new_axes.
        See Axes.check_flatten for a description of permitted values of new_axes.

        Arguments:
            new_axes: The Axes of the flattened tensor description.

        Returns:
            The reshaped tensor description.
        """
        new_axes = Axes(new_axes)
        assert Axes.check_flatten(self.axes, new_axes)

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
            name=name
        )

    def unflatten(self, new_axes, name=None):
        """
        Unflattens a tensor description to give it the Axes in new_axes.
        See Axes.check_unflatten for a description of the permitted values of
        new_axes

        Arguments:
            new_axes: The Axes of the unflattened TensorDescription.

        Returns:
            The unflattened tensor description.
        """
        new_axes = Axes(new_axes)
        assert Axes.check_unflatten(self.axes, new_axes)

        new_strides = []
        new_sizes = []
        idx = 0
        for axis, fst, fsz in zip(self.axes, self.full_strides, self.full_sizes):
            if axis == new_axes[idx]:
                new_strides.append(fst)
                new_sizes.append(fsz)
                idx += 1
            else:
                new_strides.extend(fst)
                new_sizes.extend(fsz)
                idx += len(axis.axes)

        return TensorDescription(
            new_axes,
            base=self.base,
            dtype=self.dtype,
            full_strides=new_strides,
            full_sizes=new_sizes,
            offset=self.offset,
            name=name
        )

    def transpose(self, name=None):
        """
        Reverses the axes of the tensor description.

        Retuns:
            A tensor description with the axes reversed.
        """
        new_axes = reversed(self.axes)
        full_sizes = reversed(self.full_sizes)
        full_strides = reversed(self.full_strides)
        return TensorDescription(Axes(new_axes),
                                 base=self.base,
                                 dtype=self.dtype,
                                 full_strides=tuple(full_strides),
                                 full_sizes=tuple(full_sizes),
                                 offset=self.offset,
                                 name=name)

    def broadcast(self, new_axes, name=None):
        """
        Adds axes to a tensor description to give it a new shape.
        See Axes.check_broadcast for a description of the permitted
        transformations.

        Arguments:
            new_axes: The axes of the broadcasted tensor description.

        Returns:
            TensorDescription: The broadcasted tensor description.
        """
        Axes.check_broadcast(self.axes, new_axes)
        return self.reorder_and_broadcast(new_axes, name)

    def reorder(self, new_axes, name=None):
        """
        Shuffles axes of a tensor to give it a new shape. The axes of
        this tensor description and new_axes must have the same elements.

        Arguments:
            new_axes: The axes of the reordered tensor.

        Returns:
            TensorDescription: The reordered tensor description.
        """
        Axes.same_elems(self.axes, new_axes)
        return self.reorder_and_broadcast(new_axes, name)

    def reorder_and_broadcast(self, new_axes, name):
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
                idx = Axes.find_axis(self.axes, axis)
                new_strides.append(self.full_strides[idx])
                new_sizes.append(self.full_sizes[idx])
            elif isinstance(axis, FlattenedAxis):
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
            name=name
        )

    def cast(self, new_axes, name=None):
        """
        Return a tensor desciption for a view of the tensor.

        Arguments:
            new_axes: The axes for the view.

        Returns:
            The tensor description.

        """
        return TensorDescription(
            new_axes,
            base=self.base,
            dtype=self.dtype,
            full_strides=self.full_strides,
            full_sizes=self.full_sizes,
            offset=self.offset,
            name=name
        )

    def slice(self, slices, new_axes, name=None):
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
                'in a slice operation, the number of axes pass in to '
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
                _check_sliced_axis_length(s, axis, new_axis)

                start, stop, step = s.indices(axis.length)

                full_strides.append(stride * step)
                full_sizes.append(size)

                idx = start
            else:
                # this is a simple integer slice, ex: y = x[1]
                idx = s

            # TODO: write a test that fails if abs() is removed
            offset += idx * abs(stride)

        return TensorDescription(new_axes,
                                 base=self.base,
                                 dtype=self.dtype,
                                 full_strides=tuple(full_strides),
                                 full_sizes=tuple(full_sizes),
                                 offset=offset,
                                 name=name)

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
    def base(self):
        """The viewed tensor description or None if not a view."""
        return self.__base or self

    @property
    def buffer(self):
        """The description of the underlying storage."""
        return self.base.__buffer

    @buffer.setter
    def buffer(self, value):
        """
        Sets the backend-specific memory to be used by the tensor.

        Arguments:
          value: the buffer to use

        Returns:
        """
        self.base.__buffer = value

    @property
    def register(self):
        return self.base.__register

    @register.setter
    def register(self, value):
        self.base.__register = value

    @property
    def value(self):
        """A device handle to the value."""
        return self.__value

    def is_base(self):
        """This tensor provides its own storage."""
        return self.__base is None

    def initialize(self, transformer):
        """Called by transformer to set up value."""
        assert self.__value is None
        self.transformer = transformer
        # If the TensorDescription requires heap storage
        if self.buffer is not None:
            if self.buffer.data is None:
                self.buffer.data = self.transformer.device_buffer_storage(
                    self.buffer.size, self.dtype, self.name
                )
            self.__value = self.buffer.data.device_tensor(self)
