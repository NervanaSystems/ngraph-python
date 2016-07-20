from __future__ import division
from future.utils import with_metaclass

from abc import ABCMeta
import collections
import operator
import types
import weakref

import numpy as np

from geon.backends.graph.names import NameableValue
from geon.backends.graph.environment import get_current_environment


class Axis(with_metaclass(ABCMeta, NameableValue)):
    def __init__(self, length, **kargs):
        super(Axis, self).__init__(**kargs)
        self.__length = length

    def __getitem__(self, key):
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
        return '{axis}[{idx}])'.format(axis=self.axis, idx=self.idx)

TensorAxisInfo = collections.namedtuple('TensorAxisInfo', ['length', 'stride'])


def canonicalize(seq):
    elems = []
    for x in seq:
        if isinstance(x, AxesAxis):
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
                x = AxesAxis(Axes(*x))
        elems.append(x)
    return elems


def no_duplicates(arr):
    s = set()
    for x in enumerate(arr):
        if x in s:
            return False
        s.add(x)
    return True


class Axes(tuple):
    def __new__(cls, *seq):
        if len(seq) > 0 and isinstance(seq[0], types.GeneratorType):
            assert len(seq) == 1
            seq = tuple(seq[0])
        seq = canonicalize(seq)
        assert all([isinstance(x, Axis) for x in seq])
        return tuple.__new__(cls, seq)

    @property
    def full_lengths(self):
        return tuple(x.axes.unreduced_lengths if isinstance(x, AxesAxis)
                     else x.length for x in self)

    @property
    def lengths(self):
        return tuple(x.length for x in self)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Axes(*super(Axes, self).__getitem__(item))
        else:
            return super(Axes, self).__getitem__(item)

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

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
        for i, x in enumerate(self):
            s += repr(x)
            s += ', '
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
                a = a.as_axis_ids()
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


class AxesAxis(Axis):
    def __init__(self, axes, **kargs):
        assert isinstance(axes, Axes)
        length = reduce(operator.mul, axes.lengths, 1)
        super(AxesAxis, self).__init__(length=length, **kargs)
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
        s = 'AxesAxis('
        for i, x in enumerate(self.axes):
            s += repr(x)
            s += ', '
        s += ')'
        return s


def reduce_strides(strides):
    return tuple(int(reduce_nested(elem, float('inf'), min))
                 for elem in strides)


class TensorDescription(object):
    """Axes information about an allocated tensor"""
    def __init__(self, axes, dtype=np.dtype(np.float32), full_shape=None,
                 buffer=None, value=None, full_strides=None, full_sizes=None,
                 offset=0, **kargs):
        super(TensorDescription, self).__init__(**kargs)
        # TODO: get the default type from the backend. May not always be numpy.
        # TODO: support flattening, unflattening, other complex reshapes
        axes = Axes(*axes)
        self.value = value
        self.__buffer = buffer
        self.dtype = dtype
        self.axes = axes
        self.offset = offset
        self.ndim = len(self.axes)
        self.views = weakref.WeakSet()
        self.full_shape = full_shape if full_shape is not None \
            else self.axes.full_lengths
        self.full_sizes = full_sizes if full_sizes is not None \
            else self.axes.full_lengths

        if buffer is not None:
            buffer.views.add(self)

        if full_strides is None:
            # TODO: deduce strides of nested axes.
            full_strides = []
            stride = self.dtype.itemsize
            for axis, full_size in reversed(zip(self.axes, self.full_sizes)):
                assert not isinstance(axis, AxesAxis)
                full_strides.append(stride)
                stride *= full_size
            self.full_strides = tuple(reversed(full_strides))
        else:
            self.full_strides = full_strides

        assert len(self.full_shape) == self.ndim, \
            "Shape must have same number of dimensions as axes"
        assert len(self.full_sizes) == self.ndim, \
            "Sizes must have same number of dimensions as axes"
        assert len(self.full_strides) == self.ndim, \
            "Strides must have same number of dimensions as axes"

    def __getitem__(self, item):
        assert isinstance(item, collections.Iterable)
        assert len(item) == self.ndim

        offset = self.offset
        for idx, axis, length, stride in \
                zip(item, self.axes, self.shape, self.strides):
            assert 0 <= idx and idx < length
            offset = offset + idx * stride
        return offset

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
            if isinstance(axis, AxesAxis):
                poss = []
                for sub in axis.axes:
                    assert not isinstance(sub, AxesAxis)
                    poss.append(get_old_axis(sub))
                old_poss.append(tuple(poss))
            else:
                old_poss.append(get_old_axis(axis))

        return old_poss

    def split_reduce_at(self, div_point):
        def pos_tup(lower, upper):
            if lower == upper - 1:
                return lower
            else:
                return tuple(range(lower, upper))
        if div_point == 0 or div_point == self.ndim:
            new_axes = Axes(AxesAxis(self.axes))
            old_poss = (pos_tup(0, self.ndim),)
        else:
            new_axes = Axes(
                AxesAxis(self.axes[:div_point]),
                AxesAxis(self.axes[div_point:])
            )
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

    def dot_reaxe_right(self, red_axis_ids):
        old_axis_ids = self.axes.as_axis_ids()
        idx = AxisIDTuple.find(red_axis_ids, old_axis_ids)
        axis_ids = red_axis_ids + old_axis_ids[:idx]\
            + old_axis_ids[idx + len(red_axis_ids):]
        div_point = len(red_axis_ids)
        return self.reaxe_with_axis_ids(axis_ids).split_reduce_at(div_point)

    def reaxe(self, new_axes, broadcast=True):
        new_axes = Axes(*new_axes)
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
                                         broadcast=False)

    def reaxe_with_positions(self, new_axes, old_poss, broadcast=True):
        assert len(new_axes) == len(old_poss)

        full_shape = []
        full_sizes = []
        full_strides = []

        def old_info(axis, old_pos):
            if old_pos == -1:
                full_length = axis.axes.full_lengths\
                    if isinstance(axis, AxesAxis) else axis.length
                return full_length, full_length, 0
            else:
                return self.full_shape[old_pos],\
                    self.full_sizes[old_pos], self.full_strides[old_pos]

        for axis, old_pos in zip(new_axes, old_poss):
            if isinstance(axis, AxesAxis):
                sub_shape = []
                sub_sizes = []
                sub_strides = []
                for sub, sub_pos in zip(axis.axes, old_pos):
                    assert not isinstance(sub, AxesAxis)
                    fsh, fsi, fst = old_info(sub, sub_pos)
                    sub_shape.append(fsh)
                    sub_sizes.append(fsi)
                    sub_strides.append(fst)
                full_shape.append(tuple(sub_shape))
                full_sizes.append(tuple(sub_sizes))
                full_strides.append(tuple(sub_strides))
            else:
                fsh, fsi, fst = old_info(axis, old_pos)
                full_shape.append(fsh)
                full_sizes.append(fsi)
                full_strides.append(fst)
        return TensorDescription(new_axes, dtype=self.dtype,
                                 full_shape=tuple(full_shape),
                                 full_strides=tuple(full_strides),
                                 full_sizes=tuple(full_sizes),
                                 offset=self.offset,
                                 buffer=self.buffer)

    def slice(self, slices):
        assert len(slices) == self.ndim
        base_index = []
        strides = []
        axes = []
        shape = []

        for s, stride, length, axis in \
                zip(slices, self.strides, self.shape, self.axes):
            if isinstance(s, slice):
                start, stop, step = slice.indices(length)
                base_index.append(start)
                strides.append(stride * step)
                axes.append(axis)
                shape.append((stop - start) // step)
            else:
                base_index.append(s)

        offset = self[base_index]

        return TensorDescription(axes, dtype=self.dtype,
                                 shape=shape, strides=strides, offset=offset,
                                 buffer=self.buffer)

    @property
    def strides(self):
        return reduce_strides(self.full_strides)

    @property
    def shape(self):
        return tuple(reduce_nested(_, 1, operator.mul)
                     for _ in self.full_shape)

    @property
    def sizes(self):
        return tuple(reduce_nested(_, 1, operator.mul)
                     for _ in self.full_sizes)

    @property
    def buffer(self):
        return self.__buffer or self


def get_batch_axes(default=()):
    environment = get_current_environment()
    if environment is None:
        return default
    return environment.get_value('batch_axes', default)


def set_batch_axes(axes):
    get_current_environment()['batch_axes'] = axes


def get_phase_axes(default=()):
    environment = get_current_environment()
    if environment is None:
        return default
    return environment.get_value('phase_axes', default)


def set_phase_axes(axes):
    get_current_environment()['phase_axes'] = axes
