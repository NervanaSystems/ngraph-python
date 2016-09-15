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
import numbers
import operator
import weakref
from functools import reduce, wraps

import numpy as np
import types
from abc import ABCMeta
from builtins import object, map, range, zip
from future.utils import with_metaclass

from ngraph.util.names import NameableValue


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
    programming langauges, allowing objects to interact only when
    they are permitted to do so in advance. In symbolic computation,
    this prevents interference between axes that happen to have the
    same lengths but are logically distinct, e.g. if the number of
    training examples and the number of input features are both 50.

    TODO: Please add to the list...

    Arguments:
        length: The length of the axis.
        batch: Whether the axis is a batch axis.
        recurrent: Whether the axis is a recurrent axis.

    Attributes:
        length: The length of the axis.

    """
    def __init__(self, length=None, batch=False, recurrent=False, **kwargs):
        super(Axis, self).__init__(**kwargs)
        self.__length = length
        self.batch = batch
        self.recurrent = recurrent

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

    def __repr__(self):
        return 'Axis({name}: {length})'.format(name=self.name, length=self.length)


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
    Returns whether there are duplicates in a list. The elements
    of the list should be hashable.

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
                    x = FlattenedAxis(Axes(convert(x)))
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

        assert no_duplicates(axes)
        self._axes = axes

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
        return Axes.append(self, Axes(other))

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

    @staticmethod
    @with_args_as_axes
    def append(axes1, axes2):
        """
        Returns the union of the elements, leaving out duplicate Axes.

        Arguments:
            axes1: first axes to append
            axes2: second axes to append

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
        return Axes((axis for axis in axes1 if axis not in axes2))

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
            (in_axes + out_axes) -
            Axes.intersect(in_axes, out_axes)
        )

    @staticmethod
    @with_args_as_axes
    def find(axes, sub_axes):
        """
        Attempts to locate a subsequence of Axes (subaxes) in axes.

        Arguments:
            axes: The superset of Axes.
            subaxes: Axes to search for.

        Returns:
            int: The index at which the subsequence subaxes occurs in
            axes.
        """
        assert isinstance(axes, Axes) and isinstance(sub_axes, Axes)
        for i in range(len(axes) - len(sub_axes) + 1):
            if axes[i:i + len(sub_axes)] == sub_axes:
                return i
        raise ValueError('Could not find subaxes')

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
    @with_args_as_axes
    def unflatten(axes):
        new_axes = []
        for axis in axes:
            if isinstance(axis, FlattenedAxis):
                new_axes.extend(axis.axes)
            else:
                new_axes.append(axis)
        return Axes(new_axes)

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


def reduce_nested(elem, agg, func):
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
            agg = reduce_nested(sub, agg, func)
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
        assert isinstance(axes, Axes)
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
    return tuple(int(reduce_nested(elem, float('inf'), min))
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
    if isinstance(axis, FlattenedAxis):
        return _make_strides(inner_size, axis.axes, fsz)
    else:
        stride = inner_size
        inner_size *= fsz
        return inner_size, stride


def _make_strides(inner_size, axes, full_sizes):
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
                 dtype=np.dtype(np.float32),
                 full_strides=None, full_sizes=None, offset=0,
                 **kwargs):
        super(TensorDescription, self).__init__(**kwargs)
        # TODO: get the default type from the backend. May not always be numpy.
        # TODO: support flattening, unflattening, other complex reshapes
        axes = Axes(axes)
        self.axes = axes
        self.transformer = None
        self.__casts = weakref.WeakValueDictionary()
        self.__slices = weakref.WeakValueDictionary()
        self.__value = None
        self.__buffer = None
        self.__base = base
        self.dtype = dtype
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

    def split_reduce_at(self, div_point):
        """
        Collapses a tensor description into two dimensions, flattening
        the axes before and after the index div_point.

        E.g. (C, D, E, F) with div_point 2 becomes ((C, D), (E, F)),
        that is two flattened axes.

        If div_point is such that one dimension has no axis in it, an Axes with
        only 1 Axis will be returned.

        Arguments:
          div_point: The index at which we separate the axes.

        Returns:
            The reshaped tensor description.
        """
        if div_point == 0 or div_point == self.ndim:
            # if div_point has us putting all of the axes into one Axes, just
            # make one FlattenedAxis instead.
            # raise ValueError(div_point)
            new_axes = Axes([FlattenedAxis(self.axes)])
            new_strides = (self.full_strides,)
            new_sizes = (self.full_sizes,)
        else:
            new_axes = Axes((
                FlattenedAxis(self.axes[:div_point]),
                FlattenedAxis(self.axes[div_point:])
            ))
            new_strides = (
                self.full_strides[:div_point],
                self.full_strides[div_point:]
            )
            new_sizes = (
                self.full_sizes[:div_point],
                self.full_sizes[div_point:]
            )

        return TensorDescription(
            new_axes,
            base=self.base,
            full_strides=new_strides,
            full_sizes=new_sizes,
            offset=self.offset
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
        return TensorDescription(Axes(new_axes),
                                 base=self.base,
                                 dtype=self.dtype,
                                 full_strides=tuple(full_strides),
                                 full_sizes=tuple(full_sizes),
                                 offset=self.offset)

    def unflatten(self):
        new_strides = []
        new_sizes = []
        for axis, fst, fsz in\
                zip(self.axes, self.full_strides, self.full_sizes):
            if isinstance(axis, FlattenedAxis):
                assert isinstance(fst, tuple)\
                    and isinstance(fsz, tuple)
                new_strides.extend(fst)
                new_sizes.extend(fsz)
            else:
                new_strides.append(fst)
                new_sizes.append(fsz)

        return TensorDescription(
            Axes.unflatten(self.axes),
            base=self.base,
            dtype=self.dtype,
            full_strides=new_strides,
            full_sizes=new_sizes,
            offset=self.offset
        )

    def broadcast(self, new_axes):
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
            offset=self.offset
        )

    def cast(self, new_axes):
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
            offset=self.offset
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
                                 offset=offset)

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
        return tuple(reduce_nested(_, 1, operator.mul)
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
