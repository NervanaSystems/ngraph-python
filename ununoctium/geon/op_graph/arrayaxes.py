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
from builtins import object, map, range, zip
from future.utils import with_metaclass
from functools import reduce

from abc import ABCMeta
import collections
import operator
import types
import weakref

import numpy as np

from geon.op_graph.names import NameableValue
from geon.backends.graph.environment import get_current_environment


class Axis(with_metaclass(ABCMeta, NameableValue)):

    def __init__(self, length, **kargs):
        super(Axis, self).__init__(**kargs)
        self.__length = length

    def axis_id(self, key):
        return AxisID(self, key)

    @property
    def length(self):
        return self.__length

    def __repr__(self):
        return 'Axis({name})'.format(name=self.name)


class AxisVar(Axis):
    """
    Like an axis, except the length comes from the environment.
    """

    def __init__(self, length=None, **kargs):
        super(AxisVar, self).__init__(length=-1, **kargs)
        if length is not None:
            self.length = length

    @property
    def length(self):
        return get_current_environment()[self]

    @length.setter
    def length(self, item):
        get_current_environment()[self] = item

    def __repr__(self):
        return 'AxisVar({name})'.format(name=self.name)


class NumericAxis(Axis):
    cache = {}

    def __new__(cls, length=None, **kargs):
        if length in NumericAxis.cache:
            return NumericAxis.cache[length]

        axis = super(NumericAxis, cls).__new__(cls, **kargs)
        NumericAxis.cache[length] = axis

        return axis

    def __init__(self, length=None, **kargs):
        super(NumericAxis, self).__init__(length=length, **kargs)

    def __repr__(self):
        return 'NumericAxis({length})'.format(length=self.length)


class AxisID(object):

    def __init__(self, axis, idx, **kargs):
        assert isinstance(idx, int)
        super(AxisID, self).__init__(**kargs)
        self.axis = axis
        self.idx = idx

    def __eq__(self, other):
        return isinstance(other, AxisID) \
            and self.axis == other.axis and self.idx == other.idx

    def __hash__(self):
        return hash(self.axis) + hash(self.idx)

    def __repr__(self):
        return '{axis}[{idx}]'.format(axis=self.axis, idx=self.idx)


def canonicalize(seq):
    elems = []
    for x in seq:
        if isinstance(x, FlattenedAxis):
            if x.empty:
                continue
            elif x.single:
                x = x.axes[0]
        elif isinstance(x, collections.Iterable):
            x = canonicalize(x)
            if len(x) == 0:
                continue
            elif len(x) == 1:
                x = x[0]
            else:
                x = FlattenedAxis(Axes(x))
        elems.append(x)
    return elems


def no_duplicates(arr):
    s = set()
    for x in enumerate(arr):
        if x in s:
            return False
        s.add(x)
    return True


class Axes(object):

    def __init__(self, axes=None):
        if axes is None:
            axes = []
        elif isinstance(axes, Axis):
            axes = [axes]
        elif isinstance(axes, types.GeneratorType):
            axes = tuple(axes)
        elif isinstance(axes, (list, tuple)) and not isinstance(axes, Axes):
            axes = tuple(axes)

        axes = canonicalize(axes)

        for x in axes:
            if not isinstance(x, Axis):
                raise ValueError((
                    'tried to initialize an Axes with object type '
                    '{found_type}.  all values should be an instance '
                    'of a type which inherits from Axis.'
                ).format(
                    found_type=type(x),
                ))

        self._axes = axes

    @property
    def full_lengths(self):
        return tuple(x.axes.full_lengths if isinstance(x, FlattenedAxis)
                     else x.length for x in self)

    @property
    def lengths(self):
        return tuple(x.length for x in self)

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
        if isinstance(other, Axes):
            other = other.as_axis_ids()
        return (self.as_axis_ids() + other).as_axes()

    def __sub__(self, other):
        if isinstance(other, Axes):
            other = other.as_axis_ids()
        return (self.as_axis_ids() - other).as_axes()

    def __eq__(self, other):
        if not isinstance(other, Axes):
            raise ValueError((
                'other must be of type Axes, found type {}'
            ).format(type(other)))

        return self._axes.__eq__(other._axes)

    def __ne__(self, other):
        return not self == other

    def concat(self, other):
        return Axes(tuple(self) + tuple(other))

    def index(self, axis):
        return self._axes.index(axis)

    # TODO: delete this method, the size should come from the tensor
    @property
    def size(self):
        size = 1
        for x in self:
            size *= x.length
        return size

    def as_axis_ids(self):
        m = collections.defaultdict(int)
        elems = []

        for x in self:
            index = m[x]
            m[x] = index + 1
            elems.append(AxisID(x, index))

        return AxisIDTuple(*elems)

    def __repr__(self):
        s = 'Axes('
        for x in self:
            s += repr(x) + ','
        s += ')'
        return s


# func(agg, elem) -> agg
def reduce_nested(elem, agg, func):
    if isinstance(elem, collections.Iterable):
        for sub in elem:
            agg = reduce_nested(sub, agg, func)
        return agg
    else:
        return func(agg, elem)


def with_axes_as_axis_ids(f):
    def wrapper(*args):
        new_args = []
        for a in args:
            if isinstance(a, Axes):
                a = Axes(a).as_axis_ids()
            new_args.append(a)
        return f(*new_args)
    return wrapper


class AxisIDTuple(tuple):

    def __new__(cls, *seq):
        if len(seq) > 0 and isinstance(seq[0], types.GeneratorType):
            assert len(seq) == 1
            seq = tuple(seq[0])
        seq = [x[0] if isinstance(x, Axis) else x for x in seq]
        assert all([isinstance(x, AxisID) for x in seq])
        return tuple.__new__(cls, seq)

    def as_axes(self):
        return Axes(x.axis for x in self)

    @staticmethod
    @with_axes_as_axis_ids
    def sub(at1, at2):
        assert isinstance(at1, AxisIDTuple) and isinstance(at2, AxisIDTuple)
        return AxisIDTuple(_ for _ in at1 if _ not in at2)

    @staticmethod
    @with_axes_as_axis_ids
    def intersect(at1, at2):
        assert isinstance(at1, AxisIDTuple) and isinstance(at2, AxisIDTuple)
        return AxisIDTuple(_ for _ in at1 if _ in at2)

    @staticmethod
    @with_axes_as_axis_ids
    def append(*at_list):
        assert all([isinstance(at, AxisIDTuple) for at in at_list])
        elems = []
        for at in at_list:
            for x in at:
                if x not in elems:
                    elems.append(x)
        return AxisIDTuple(*elems)

    @staticmethod
    @with_axes_as_axis_ids
    def find(subaxes, axes):
        assert isinstance(subaxes, AxisIDTuple)\
            and isinstance(axes, AxisIDTuple)
        for i in range(len(axes)):
            if axes[i:i + len(subaxes)] == subaxes:
                return i
        raise ValueError('Could not find subaxes')

    def __getitem__(self, item):
        if isinstance(item, slice):
            return AxisIDTuple(*super(AxisIDTuple, self).__getitem__(item))
        else:
            return super(AxisIDTuple, self).__getitem__(item)

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def __add__(self, other):
        return AxisIDTuple.append(self, other)

    def __sub__(self, other):
        return AxisIDTuple.sub(self, other)

    def __repr__(self):
        s = 'AxisIDTuple('
        for i, x in enumerate(self):
            s += repr(x)
            s += ', '
        s += ')'
        return s


class FlattenedAxis(Axis):
    """ A FlattenedAxis has length which is the product of the lengths of all
    Axis in the axes.  The original Axes object is stored so that we can later
    unflatten this Axis back to its original component Axis. """

    def __init__(self, axes, **kargs):
        assert isinstance(axes, Axes)
        length = reduce(operator.mul, axes.lengths, 1)
        super(FlattenedAxis, self).__init__(length=length, **kargs)
        self.__axes = axes

    @property
    def empty(self):
        return len(self.__axes) == 0

    @property
    def single(self):
        return len(self.__axes) == 1

    @property
    def axes(self):
        return self.__axes

    def __repr__(self):
        s = 'FlattenedAxis('
        for i, x in enumerate(self.axes):
            s += repr(x)
            s += ', '
        s += ')'
        return s


def reduce_strides(strides):
    return tuple(int(reduce_nested(elem, float('inf'), min))
                 for elem in strides)


class TensorDescription(NameableValue):
    """
    Description of a tensor that will be allocated in hardware.
    Names the tensor's dimensions with axes and holds pointers to the
    buffer allocated by the analysis and the backend tensor value
    (e.g. a numpy or gpu tensor).
    """

    def __init__(self, axes, transformer, base=None,
                 dtype=np.dtype(np.float32),
                 full_strides=None, full_sizes=None, offset=0,
                 **kargs):
        super(TensorDescription, self).__init__(**kargs)
        # TODO: get the default type from the backend. May not always be numpy.
        # TODO: support flattening, unflattening, other complex reshapes
        axes = Axes(axes)
        self.axes = axes
        self.transformer = transformer
        self.__value = None
        self.__buffer = None
        self.__base = base
        self.dtype = dtype
        self.offset = offset
        self.ndim = len(self.axes)
        self.views = weakref.WeakSet()
        self.__read_only = False
        self.full_sizes = full_sizes if full_sizes is not None \
            else self.axes.full_lengths

        if base is not None:
            base.views.add(self)
        transformer.tds.add(self)

        if full_strides is None:
            # TODO: deduce strides of nested axes.
            full_strides = []
            stride = self.dtype.itemsize
            for axis, full_size in reversed(
                    list(zip(self.axes, self.full_sizes))):
                assert not isinstance(axis, FlattenedAxis)
                full_strides.append(stride)
                stride *= full_size
            self.full_strides = tuple(reversed(full_strides))
        else:
            self.full_strides = full_strides

        assert len(self.full_sizes) == self.ndim, \
            "Sizes must have same number of dimensions as axes"
        assert len(self.full_strides) == self.ndim, \
            "Strides must have same number of dimensions as axes"

    def try_guess_positions(self, new_axes):
        # Supports broadcast and combining one level of axes
        # Does not support unrolling
        old_poss = []

        used_set = set()

        def get_old_axis(new_axis):
            for i, axis in enumerate(self.axes):
                if i not in used_set and axis == new_axis:
                    used_set.add(i)
                    return i
            else:
                return -1

        for axis in new_axes:
            old_pos = get_old_axis(axis)
            if old_pos == -1 and isinstance(axis, FlattenedAxis):
                poss = []
                for sub in axis.axes:
                    assert not isinstance(sub, FlattenedAxis)
                    poss.append(get_old_axis(sub))
                old_poss.append(tuple(poss))
            else:
                old_poss.append(old_pos)
        return old_poss

    def split_reduce_at(self, div_point):
        def pos_tup(lower, upper):
            if lower == upper - 1:
                return lower
            else:
                return tuple(range(lower, upper))
        if div_point == 0 or div_point == self.ndim:
            new_axes = Axes([FlattenedAxis(self.axes)])
            old_poss = (pos_tup(0, self.ndim),)
        else:
            new_axes = Axes([
                FlattenedAxis(self.axes[:div_point]),
                FlattenedAxis(self.axes[div_point:])
            ])
            old_poss = (
                pos_tup(0, div_point),
                pos_tup(div_point, self.ndim)
            )
        return self.reaxe_with_positions(new_axes, old_poss)

    def dot_reaxe_left(self, red_axis_ids):
        old_axis_ids = self.axes.as_axis_ids()
        idx = AxisIDTuple.find(red_axis_ids, old_axis_ids)
        axis_ids = old_axis_ids[:idx]\
            + old_axis_ids[idx + len(red_axis_ids):]\
            + red_axis_ids
        div_point = len(old_axis_ids) - len(red_axis_ids)
        return self.reaxe_with_axis_ids(axis_ids).split_reduce_at(div_point)

    # This function is symmetric to dot_reaxe_left unless forward_axis
    # ids is specified. It then attempts to rename the reduction axis using
    # the mapping from the forward axis ids to the current axis ids.
    # In the case of backpropagation, this helps preserve the axis id numbering
    # of the original output, which is necessary if the derivative is to be
    # projected onto the input correctly.
    def dot_reaxe_right(self, red_axis_ids, forward_axis_ids=None):
        old_axis_ids = self.axes.as_axis_ids()
        if forward_axis_ids:
            trans = dict(list(zip(forward_axis_ids, old_axis_ids)))

            def trans_func(x):
                if x in trans:
                    return trans[x]
                else:
                    return x

            red_axis_ids = AxisIDTuple(*list(map(trans_func, red_axis_ids)))
        idx = AxisIDTuple.find(red_axis_ids, old_axis_ids)
        axis_ids = red_axis_ids + old_axis_ids[:idx]\
            + old_axis_ids[idx + len(red_axis_ids):]
        div_point = len(red_axis_ids)
        return self.reaxe_with_axis_ids(axis_ids).split_reduce_at(div_point)

    def reaxe(self, new_axes, broadcast=True):
        new_axes = Axes(new_axes)
        old_poss = self.try_guess_positions(new_axes)
        return self.reaxe_with_positions(new_axes, old_poss, broadcast)

    def reaxe_with_axis_ids_positions(self, new_axis_id_tuple):
        old_axis_ids = self.axes.as_axis_ids()

        old_poss = []
        for axis_id in new_axis_id_tuple:
            for i, old_axis_id in enumerate(old_axis_ids):
                if axis_id == old_axis_id:
                    old_poss.append(i)
        return old_poss

    def reaxe_with_axis_ids(self, new_axis_id_tuple):
        # This function does not allow any unrolling of axes
        # The argument is a tuple of axis ids.
        # The indices of the axis ids refer to the existing order of axes
        old_poss = self.reaxe_with_axis_ids_positions(new_axis_id_tuple)
        return self.reaxe_with_positions(new_axes=new_axis_id_tuple.as_axes(),
                                         old_poss=old_poss,
                                         broadcast=True)

    def reaxe_with_dummy_axis(self, dummy_axis, dim=-1):
        if dim == -1:
            dim = len(self.axes)
        new_axes = self.axes[:dim]\
            .concat(Axes(dummy_axis,)).concat(self.axes[dim:])
        old_poss = list(range(dim)) + [-1] + list(range(dim, len(self.axes)))
        return self.reaxe_with_positions(new_axes=new_axes,
                                         old_poss=old_poss,
                                         broadcast=True)

    def reaxe_with_positions(self, new_axes, old_poss, broadcast=True):
        assert len(new_axes) == len(old_poss)

        full_sizes = []
        full_strides = []

        def old_info(axis, old_pos):
            if old_pos == -1:
                full_length = axis.axes.full_lengths\
                    if isinstance(axis, FlattenedAxis) else axis.length
                return full_length, 0
            else:
                return self.full_sizes[old_pos], self.full_strides[old_pos]

        for axis, old_pos in zip(new_axes, old_poss):
            if isinstance(axis, FlattenedAxis):
                sub_sizes = []
                sub_strides = []
                for sub, sub_pos in zip(axis.axes, old_pos):
                    assert not isinstance(sub, FlattenedAxis)
                    fsi, fst = old_info(sub, sub_pos)
                    sub_sizes.append(fsi)
                    sub_strides.append(fst)
                full_sizes.append(tuple(sub_sizes))
                full_strides.append(tuple(sub_strides))
            else:
                fsi, fst = old_info(axis, old_pos)
                full_sizes.append(fsi)
                full_strides.append(fst)

        new_axes, full_strides, full_sizes\
            = self.maybe_collapse_numerics(
                new_axes, full_strides, full_sizes
            )

        return TensorDescription(new_axes,
                                 self.transformer,
                                 base=self.base,
                                 dtype=self.dtype,
                                 full_strides=tuple(full_strides),
                                 full_sizes=tuple(full_sizes),
                                 offset=self.offset)

    def maybe_collapse_numerics(self, axes, full_strides, full_sizes):
        def all_numeric(axes):
            return all([isinstance(axis, NumericAxis) for axis in axes])

        new_axes = []
        new_strides = []
        new_sizes = []
        for axis, st, si in\
                zip(axes, full_strides, full_sizes):
            if isinstance(axis, FlattenedAxis) and all_numeric(axis.axes):
                new_axes.append(NumericAxis(reduce_nested(
                    axis.axes.lengths, 1, operator.mul
                )))
                new_strides.append(int(reduce_nested(st, float('inf'), min)))
                new_sizes.append(reduce_nested(si, 1, operator.mul))
            else:
                new_axes.append(axis)
                new_strides.append(st)
                new_sizes.append(si)
        return Axes(new_axes), tuple(new_strides), tuple(new_sizes)

    def slice(self, slices, new_axes):
        slices = list(slices)
        while len(slices) < self.ndim:
            slices.append(slice(None))
        offset = self.offset
        full_strides = []
        full_sizes = []
        new_index = 0

        for s, axis, stride, size in\
                zip(slices, self.axes, self.strides, self.sizes):
            if isinstance(s, slice):
                new_axis = new_axes[new_index]
                new_index += 1

                start, stop, step = s.indices(axis.length)
                assert step == 1
                assert stop - start == new_axis.length,\
                    'Axis %s has a length of %s but the acutal length is %s.'\
                    % (new_axis, new_axis.length, stop - start)

                full_strides.append(stride)
                full_sizes.append(size)

                idx = start
            else:
                idx = s

            offset += idx * stride
        return TensorDescription(new_axes,
                                 self.transformer,
                                 base=self.base,
                                 dtype=self.dtype,
                                 full_strides=tuple(full_strides),
                                 full_sizes=tuple(full_sizes),
                                 offset=offset)

    @property
    def shape(self):
        return self.axes.lengths

    @property
    def strides(self):
        return reduce_strides(self.full_strides)

    @property
    def sizes(self):
        return tuple(reduce_nested(_, 1, operator.mul)
                     for _ in self.full_sizes)

    @property
    def cache_key(self):
        return (self, 'td_values')

    @property
    def base(self):
        return self.__base or self

    @property
    def buffer(self):
        return self.base.__buffer

    @buffer.setter
    def buffer(self, value):
        self.base.__buffer = value

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, tensor):
        assert self.value.shape == tensor.shape
        self.transformer.set_item(self.value, (), tensor)

    def is_base(self):
        return self.__base is None

    def initialize(self):
        assert self.__value is None
        # If the TensorDescription requires heap storage
        if self.buffer is not None:
            if self.buffer.data is None:
                self.buffer.data = self.transformer.make_raw_buffer(
                    self.buffer.size
                )
            self.__value = self.transformer.tensor_view(self)


def with_args_as_axes(f):

    def cast(arg):
        if isinstance(arg, Axes):
            return arg
        elif isinstance(arg, AxisID):
            return arg.as_axes()
        else:
            return Axes(*arg)

    def wrapper(*args):
        return f(*(cast(arg) for arg in args))
    return wrapper


def get_batch_axes(default=Axes()):
    environment = get_current_environment()
    if environment is None:
        return default
    return environment.get_value('batch_axes', default)


@with_args_as_axes
def set_batch_axes(axes):
    get_current_environment()['batch_axes'] = axes


def get_phase_axes(default=Axes()):
    environment = get_current_environment()
    if environment is None:
        return default
    return environment.get_value('phase_axes', default)


@with_args_as_axes
def set_phase_axes(axes):
    get_current_environment()['phase_axes'] = axes


@with_args_as_axes
def sample_axes(axes):
    return axes - get_batch_axes()


@with_args_as_axes
def batch_axes(axes):
    return AxisIDTuple.intersect(
        axes.as_axis_ids(),
        get_batch_axes().as_axis_ids()
    ).as_axes()


@with_args_as_axes
def linear_map_axes(in_axes, out_axes):
    in_axis_ids, out_axis_ids = in_axes.as_axis_ids(), out_axes.as_axis_ids()
    return (
        (out_axis_ids + in_axis_ids) -
        AxisIDTuple.intersect(in_axis_ids, out_axis_ids)
    ).as_axes()
