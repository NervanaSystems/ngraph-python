from abc import ABCMeta
import collections
import numbers

import numpy as np

from geon.backends.graph.names import NameableValue
from geon.backends.graph.environment import get_current_environment


class Axis(NameableValue):
    __metaclass__ = ABCMeta

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


class AxesComp(object):
    """A Computation for computing axes"""
    def __init__(self, **kargs):
        super(AxesComp, self).__init__(**kargs)

    @staticmethod
    def as_axes(axes, **kargs):
        if isinstance(axes, AxesComp):
            return axes
        elif axes is None:
            return None
        else:
            return LiteralAxesComp(axes, **kargs)

    def evaluate(self, evaluator):
        raise NotImplementedError()

    def __add__(self, x):
        return AxesAppendComp(self, AxesComp.as_axes(x))

    def __radd__(self, x):
        return AxesAppendComp(AxesComp.as_axes(x), self)

    def __sub__(self, x):
        return AxesSubComp(self, AxesComp.as_axes(x))

    def __rsub__(self, x):
        return AxesSubComp(AxesComp.as_axes(x), self)

    def __mul__(self, x):
        return AxesIntersectComp(self, AxesComp.as_axes(x))

    def __rmul__(self, x):
        return AxesIntersectComp(AxesComp.as_axes(x), self)


class LiteralAxesComp(AxesComp):
    """Actual axes are provided"""
    def __init__(self, axes, **kargs):
        super(LiteralAxesComp, self).__init__(**kargs)
        self.axes = axes

    def evaluate(self, evaluator):
        return self.axes


class BatchAxes(AxesComp):
    """Axes used for batch"""
    def __init__(self, **kargs):
        super(BatchAxes, self).__init__(**kargs)

    def evaluate(self, evaluator):
        return evaluator.get_batch_axes()


class ValueAxesComp(AxesComp):
    """Determine axes from value computed by x"""
    def __init__(self, x, **kargs):
        super(ValueAxesComp, self).__init__(**kargs)
        self.x = x

    def evaluate(self, evaluator):
        return evaluator.get_cached_resolved_tensor_axes(self.x)


class AxesSubComp(AxesComp):
    """Result will be removal of axes in y from those in x"""
    def __init__(self, x, y, **kargs):
        super(AxesSubComp, self).__init__(**kargs)
        self.x = x
        self.y = y

    def evaluate(self, evaluator):
        x_axes = evaluator.get_resolved_axes(self.x)
        y_axes = evaluator.get_resolved_axes(self.y)
        return axes_sub(x_axes, y_axes)


def sample_axes(x, **kargs):
    return AxesSubComp(AxesComp.as_axes(x, **kargs), BatchAxes(**kargs))


def batch_axes(x, **kargs):
    return AxesAppendComp(AxesComp.as_axes(x, **kargs), BatchAxes(**kargs))


class AxesIntersectComp(AxesComp):
    def __init__(self, x, y, **kargs):
        super(AxesIntersectComp, self).__init__(**kargs)
        self.x = x
        self.y = y

    def evaluate(self, evaluator):
        x_axes = evaluator.get_resolved_axes(self.x)
        y_axes = evaluator.get_resolved_axes(self.y)
        return axes_intersect(x_axes, y_axes)


class AxesAppendComp(AxesComp):
    def __init__(self, x, y, **kargs):
        super(AxesAppendComp, self).__init__(**kargs)
        self.x = x
        self.y = y

    def evaluate(self, evaluator):
        x_axes = evaluator.get_resolved_axes(self.x)
        y_axes = evaluator.get_resolved_axes(self.y)
        return axes_append(x_axes, y_axes)

# This one should also work, but there are some bugs in axes/dot
def linear_map_axesa(in_axes, out_axes):
    return AxesSubComp(AxesAppendComp(in_axes, out_axes),
                       AxesIntersectComp(in_axes, out_axes))


def linear_map_axes(in_axes, out_axes):
    return AxesSubComp(AxesAppendComp(out_axes, in_axes),
                       AxesIntersectComp(in_axes, out_axes))


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


def axes_contains(sub_axes, super_axes):
    for axis in sub_axes:
        if axis not in super_axes:
            return False
    return True


def elementwise_axes(args_axes, out_axes=None):
    if out_axes is None:
        out_axes = args_axes

    if not axes_contains(args_axes, out_axes):
        raise ValueError('out_axes is missing axes {a}'.format(a=axes_sub(args_axes, out_axes)))

    return out_axes


def axes_replace(axes, replace, replacements):
    """Returns axes with those axes in replace replace by those in replacements"""
    ids = axis_ids(axes)
    r = dict()
    for k in ids:
        r[k] = k
    for k,v in zip(axis_ids(replace), axis_ids(replacements)):
        r[k] = v
    return [r[axis] for axis in ids]


def axes_reshape(in_axes, out_axes):
    """
    Compute the reshape shape to broadcast in to out.  Axes must be consistently ordered

    :param in_axes: Axes of the input
    :param out_axes: Axes of the output
    :return: shape argument for reshape()
    """
    result = []
    in_axes = axis_ids(in_axes)
    out_axes = axis_ids(out_axes)

    for out_axis in out_axes:
        if out_axis in in_axes:
            result.append(out_axis.length)
        else:
            result.append(1)
    return tuple(result)


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


def flatten_shape(shape):
    s = []
    for l in shape:
        if isinstance(l, collections.Sequence):
            s += l
        else:
            s.append(l)
    return s


def reaxe(x, axes, broadcast=False):
    if isinstance(x, Scalar):
        if broadcast or axes == ():
            return x
    elif isinstance(x, AxisArray):
        return x.reaxe(axes, broadcast)

    raise ValueError("{x} has no axes".format(x=x))


def dot_axes(x_axes, y_axes, reduction_axes=None, out_axes=None):
    xy_axes = axes_intersect(x_axes, y_axes)
    x_or_y_axes = axes_append(x_axes, y_axes)
    if reduction_axes is None:
        if out_axes is None:
            reduction_axes = xy_axes
        else:
            reduction_axes = axes_sub(x_or_y_axes, out_axes)

    if out_axes is None:
        out_axes = axes_sub(x_or_y_axes, reduction_axes)

    # Common but not reduced
    s_axes = axes_sub(xy_axes, reduction_axes)

    return reduction_axes, s_axes, out_axes


def permute(array, permutation):
    old_strides = array.strides
    old_shape = array.shape
    new_strides = [old_strides[pos] for pos in permutation]
    new_shape = [old_shape[pos] for pos in permutation]
    return np.ndarray(shape=new_shape, dtype=array.dtype, buffer=array, strides=new_strides)


def invert_permutation(permutation):
    result = list(permutation)
    for i, p in enumerate(permutation):
        result[p] = i
    return result


def permutation(array):
    """The permutation to put the axes of the array in storage order (decreasing strides)"""
    t = sorted([(i, s) for i, s in enumerate(array.strides)], key=lambda p: -p[1])
    return tuple(i for i, s in t)

def shapes_compatible(shape1, shape2):
    def shape_size(shape):
        result = 1
        for l in flatten_shape(shape):
            result *= l
        return result

    return shape_size(shape1) == shape_size(shape2)


class AxisArray(np.ndarray):
    """
    Helper to keep arrays and axes paired until shaping is explicit in the graph.
    """
    __metaclass__ = ABCMeta

    def __new__(cls, axes=None, array=None, base=None, broadcast=False, dtype=None, buffer=None, offset=0,
                strides=None, order=None):
        if axes is not None:
            def canonic(x):
                if isinstance(x, collections.Iterable):
                    x = tuple(x)
                    if len(x) == 1:
                        return x[0]
                    else:
                        return x
                else:
                    return x
            axes = tuple(canonic(x) for x in axes)
        shape = tuple(axes_shape(axes))
        if array is not None:
            array = np.asarray(array)
            if not shapes_compatible(array.shape, shape):
                raise ValueError('Incompatible shapes')
            base = array.reshape(shape)
            dtype = array.dtype

        if dtype is None:
            dtype = float

        if base is not None:
            buffer = base.base
            if buffer is None:
                buffer = base
            base_strides = base.strides
            if isinstance(buffer, ObjectWithAxes):
                base_axes = buffer.axes
            elif isinstance(base, ObjectWithAxes):
                base_axes = base.axes
            else:
                base_axes = axes

            def stride(axis):
                index = base_axes.index(axis)
                if index < 0:
                    return 0
                return base_strides[index]

            strides = []
            for axis in axes:
                component_axes = flatten_axes(axis)
                if len(component_axes) == 0:
                    raise ValueError('Cannot compose empty axis set')
                axis_pos = base_axes.index(component_axes[0])
                if axis_pos == -1:
                    if not broadcast:
                        raise ValueError('Cannot reaxe with axis {a} not in axes'.format(a=axis))
                    for caxis in component_axes:
                        if base_axes.index(caxis) != -1:
                            raise ValueError('Cannot compose broadcast and non-broadcast axes')
                else:
                    for i, caxis in enumerate(component_axes):
                        if axis_pos + i != base_axes.index(caxis):
                            raise ValueError('Cannot compose non-contiguous axes')

                strides.append(stride(component_axes[-1]))

        obj = np.ndarray.__new__(cls, shape, dtype, buffer, offset, strides, order)
        obj.axes = axes
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        obj_axes = getattr(obj, 'axes', None)
        obj_shape = axes_shape(obj_axes)
        axes = []
        shape = self.shape
        if obj_shape != shape:
            for axis, s in zip(obj_axes, self.shape):
                if isinstance(axis, list):
                    pass
                if axis.length != s:
                    axis = Axis(s, like=axis)
                axes.append(axis)
        self.axes = tuple(axes)

    def reaxe(self, axes, broadcast=False):
        axes = tuple(axes)
        if axes == self.axes:
            return self

        return AxisArray(base=self, axes=axes, broadcast=broadcast, dtype=self.dtype)

    def axes_like(self, array):
        # TODO Is this reaxe?
        """
        Reshape self to conform ot array.

        :param array:
        :return: An alias of self shaped like array
        """
        return self.array_as_axes(array.axes)

    def array_as_axes(self, axes):
        # TODO Is this reaxe?
        shape = axes_reshape(self.axes, axes)

        if self.shape == shape:
            return self
        return self.reshape(shape)


class ObjectWithAxes(object):
    __metaclass__ = ABCMeta


ObjectWithAxes.register(AxisArray)


class Scalar(object):
    __metaclass__ = ABCMeta


Scalar.register(numbers.Real)
ObjectWithAxes.register(Scalar)

def axes(x):
    if isinstance(x, Scalar):
        return ()

    if isinstance(x, ObjectWithAxes):
        return x.axes
    raise ValueError("{x} has no axes".format(x=x))


def axes_generator(axes, gen_axis, base_index=None):
    index = list(base_index or [0]*len(axes))
    pos = axes.index(gen_axis)
    for i in xrange(gen_axis.length):
        index[pos] = i
        yield tuple(index)



