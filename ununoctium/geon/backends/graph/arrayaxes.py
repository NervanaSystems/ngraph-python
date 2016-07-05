from __future__ import division
from future.utils import with_metaclass

from abc import ABCMeta
import collections
import math
import weakref

import numpy as np

from geon.backends.graph.names import NameableValue
from geon.backends.graph.environment import get_current_environment


class Axis(with_metaclass(ABCMeta, NameableValue)):
    def __init__(self, length, like=None, **kargs):
        super(Axis, self).__init__(**kargs)
        self._length = length
        if like is not None:
            self.like = like.axis
        else:
            self.like = None

    def __getitem__(self, key):
        if key == self.idx:
            return self
        return AxisID(self, key)

    @property
    def length(self):
        return self._length

    @property
    def idx(self):
        return 0

    @property
    def axis(self):
        return self

    def as_axisid(self):
        return self[0]

    def __repr__(self):
        return 'Axis({name})'.format(name=self.name or self.like)


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
        return 'AxisVar({name})'.format(name=self.name or self.like)


class AxisID(object):
    def __init__(self, axis, idx, **kargs):
        super(AxisID, self).__init__(**kargs)
        self.axis = axis
        self.idx = idx

    def __getitem__(self, key):
        if key == self.axis.idx:
            return self.axis
        if key == self.idx:
            return self
        return AxisID(self.axis, key)

    def as_axisid(self):
        return self

    @property
    def length(self):
        return self.axis.length

    @length.setter
    def length(self, length):
        self.axis.length = length

    def __eq__(self, other):
        return isinstance(other, AxisID) and self.axis == other.axis and self.idx == other.idx

    def __hash__(self):
        return hash(self.axis)+hash(self.idx)

    def __repr__(self):
        return '{axis}[{idx}])'.format(axis=self.axis, idx=self.idx)


Axis.register(AxisID)


TensorAxisInfo = collections.namedtuple('TensorAxisInfo', ['length', 'stride'])


class TensorDescription(object):
    """Axes information about an allocated tensor"""

    def __init__(self, axes, dtype=np.dtype(np.float32), shape=None, sizes=None, strides=None, offset=0, verify=True,
                 buffer=None, value=None, **kargs):
        super(TensorDescription, self).__init__(**kargs)
        self.dtype = np.dtype(dtype)
        self.axes = tuple(canonicalize_axes(axes))
        self.ndim = len(self.axes)
        self.offset = offset
        self.__buffer = buffer
        self.value = value
        self.views = weakref.WeakSet()

        if buffer is not None:
            buffer.views.add(self)

        if shape is None:
            shape = tuple(axes_sizes(self.axes))
        self.shape = shape

        if sizes is None:
            self.sizes = self.shape
        else:
            self.sizes = tuple(sizes)
            assert len(self.sizes) == self.ndim, "Sizes must have same number of dimensions as axes"

        self.axes_info = collections.OrderedDict()
        if strides is None:
            strides = []
            stride = self.dtype.itemsize
            for axis, size in reversed(zip(self.axes, self.sizes)):
                if verify:
                    assert axis.length <= size, "An dimension size cannot be less than the dimension length"
                self.axes_info[axis] = TensorAxisInfo(length=axis.length, stride=stride)
                stride = stride * size
                strides.append(stride)
            self.strides = tuple(reversed(strides))
        else:
            assert len(strides) == self.ndim
            self.strides = tuple(strides)
            for axis, length, stride in zip(self.axes, axes_shape(self.axes), self.strides):
                self.axes_info[axis] = TensorAxisInfo(length=length, stride=stride)

    @property
    def buffer(self):
        return self.__buffer or self


    def __getitem__(self, item):
        if isinstance(item, Axis):
            return self.axes_info[item]

        elif isinstance(item, collections.Iterable):
            assert len(item) == self.ndim
            offset = self.offset
            for idx, axis_info in zip(item, self.axes_info):
                assert 0 <= idx and idx < axis_info.length
                offset = offset + idx * axis_info.stride
            return offset

    def reaxe(self, reaxes, broadcast=True):
        if self.axes == tuple(reaxes):
            return self

        # Format of reaxes:
        # Axis: position is an axis
        # tuple of axes: position is combination of the axes
        strides = []
        for axis in reaxes:
            component_axes = flatten_axes(axis)
            if len(component_axes) == 0:
                strides.append(0)
            else:
                try:
                    strides.append(self[component_axes[-1]].stride)
                except KeyError:
                    if not broadcast:
                        raise ValueError('Cannot reaxe with axis {a} not in axes'.format(a=axis))
                    else:
                        strides.append(0)
        shape = tuple(axes_shape(reaxes))
        return TensorDescription(reaxes, dtype=self.dtype, shape=shape, strides=strides, offset=self.offset,
                                 buffer=self.buffer)

    def slice(self, slices):
        assert len(slices) == self.ndim
        base_index = []
        shape = []
        strides = []
        axes = []
        for s, stride, length, axis in zip(slices, self.strides, self.shape, self.axes):
            if isinstance(s, slice):
                # Keep slice dimensions
                start, stop, step = slice.indices(length)
                base_index.append(start)
                strides.append(stride * step)
                shape.append((stop - start) // step)
                axes.append(axis)
            else:
                # Omit fixed dimensions
                base_index.append(s)

        offset = self[base_index]

        return TensorDescription(axes, dtype=self.dtype, shape=shape, strides=strides, offset=offset,
                                 buffer=self.buffer)


def canonicalize_axes(axes):
    def canonic(x):
        if isinstance(x, collections.Iterable):
            x = tuple(x)
            if len(x) == 1:
                return x[0]
            else:
                return x
        else:
            return x

    return tuple(canonic(x) for x in axes)


def axis_ids(axes):
    """Return a list of axes with unique axis indices"""
    result = []
    for axis in axes:
        if axis in result:
            axis = axis.axis
            while axis in result or axis in axes:
                axis = axis[axis.idx + 1]
        result.append(axis)
    return result


def find_axes_in_axes(subaxes, axes):
    subaxes = axis_ids(subaxes)
    axes = axis_ids(axes)
    if not subaxes:
        return 0
    head = subaxes[0]
    for i, axis in enumerate(axes):
        if head == axis and axes[i:i+len(subaxes)] == subaxes:
            return i
    return -1


def axes_sub(x, y):
    """Returns x with elements from y removed"""
    return [_ for _ in axis_ids(x) if _ not in axis_ids(y)]


def axes_intersect(x, y):
    """Returns intersection of x and y in x order"""
    return [_ for _ in axis_ids(x) if _ in axis_ids(y)]


def axes_append(*axes_list):
    """Returns x followed by elements of y not in x"""
    result = []
    for axes in axes_list:
        ids = axis_ids(axes)
        for axis_id in ids:
            if axis_id not in result:
                result.append(axis_id)
    return result


def axes_replace(axes, replace, replacements):
    """Returns axes with those axes in replace replace by those in replacements"""
    ids = axis_ids(axes)
    r = dict()
    for k in ids:
        r[k] = k
    for k,v in zip(axis_ids(replace), axis_ids(replacements)):
        r[k] = v
    return [r[axis] for axis in ids]


def axes_list(axes, shape_list):
    result = []
    for shape in shape_list:
        for axis, size in zip(axes, shape):
            axis.length = size
        result.append(axes)
        axes = [axis.prime() for axis in axes]
    return result


def flatten_axes(axes):
    """Return axes with all tuples expanded."""
    result = []
    def flatten1(axes):
        if isinstance(axes, collections.Sequence):
            for _ in axes:
                flatten1(_)
        else:
            result.append(axes)

    flatten1(axes)
    return result


def axes_shape(axes):
    shape = []
    for axis in axes:
        length = 1
        for caxis in flatten_axes(axis):
            length = length*caxis.length
        shape.append(length)
    return shape


def axes_size(axes):
    size = 1
    for axis in axes:
        for caxis in flatten_axes(axis):
            size *= caxis.length
    return size


def axes_sizes(axes):
    return [axis.length for axis in axes]


def output_dim(self, X, F, padding, strides, pooling=False, cafe_compatibility=False):
    """
    Compute along 1 dimension, with these sizes, what will be the output dimension.

    Arguments:
        X (int): input data dimension
        F (int): filter dimension
        padding (int): padding on each side
        strides (int): striding
        pooling (bool): flag for setting pooling layer size
    """

    if cafe_compatibility and pooling:
        size = int(math.ceil((float(X - F + 2 * padding) / strides))) + 1
        if padding > 0 and (size - 1) * strides >= X + padding:
            # decrement size if last pooling op is completely in padding
            size -= 1
    else:
        # normal neon output size determination
        size = ((X - F + 2 * padding) // strides) + 1

    if pooling and padding >= F:
        raise ValueError("Padding dim %d incompatible with filter size %d" % (padding, F))

    return size


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
