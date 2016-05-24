import abc
import numbers
import collections
import numpy as np

# This is a partial implementation of axes on top of NumPy

# TODO Track the computation so that we can interactively autodiff

# TODO Only the simpler axis cases are handled

# TODO AxisIDs are mostly missing


def axes_generator(axes, gen_axis, base_index=None):
    index = list(base_index or [0]*len(axes))
    pos = axes.index(gen_axis)
    for i in xrange(gen_axis.length):
        index[pos] = i
        yield tuple(index)

class HasAxes:
    __metaclass__ = abc.ABCMeta

    @property
    def axes(self):
        return ()


class HasScalarAxes(HasAxes):
    pass


HasScalarAxes.register(numbers.Real)


def axes(x):
    if isinstance(x, HasScalarAxes):
        return ()
    elif isinstance(x, HasAxes):
        return x.axes
    else:
        raise ValueError("{x} has no axes".format(x=x))


def reaxe(x, axes, broadcast=False):
    if isinstance(x, HasScalarAxes):
        if broadcast or axes == ():
            return x
    elif isinstance(x, HasAxes):
        return x.reaxe(axes, broadcast)

    raise ValueError("{x} has no axes".format(x=x))


def axes_shape(axes):
    shape = []
    for axis in axes:
        length = 1
        for caxis in flatten_axes(axis):
            length = length*caxis.length
        shape.append(length)
    return shape


class Axis(object):
    def __init__(self, name, length, **kargs):
        super(Axis, self).__init__(**kargs)
        self.name = name
        self.__length = length

    @property
    def length(self):
        return self.__length

    def flatten(self):
        return [self]

    def __repr__(self):
        return '{name}[{length}]'.format(name=self.name, length=self.length)

def flatten_axes(axes):
    result = []
    def flatten1(axes):
        if isinstance(axes, collections.Sequence):
            for _ in axes:
                flatten1(_)
        else:
            result.append(axes)

    flatten1(axes)
    return result


def find_axes_in_axes(subaxes, axes):
    subaxes = list(subaxes)
    axes = list(axes)
    if not subaxes:
        return 0
    head = subaxes[0]
    for i, axis in enumerate(axes):
        if head is axis and axes[i:i+len(subaxes)] == subaxes:
            return i
    return -1


def axes_sub(x, y):
    """Returns x with elements from y removed"""
    return [_ for _ in x if _ not in y]


def axes_intersect(x, y):
    """Returns intersection of x and y in x order"""
    return [_ for _ in x if _ in y]


def axes_append(*axes_list):
    """Returns x followed by elements of y not in x"""
    result = []
    for axes in axes_list:
        for axis in axes:
            if axis not in result:
                result.append(axis)
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


def elementwise_reaxe(args, out=None, out_axes=None):
    args_axes = axes_append(*(axes(arg) for arg in args))
    if out_axes is None and out is not None:
        out_axes = axes(out)

    if out is not None and axes(out) != out_axes:
        raise ValueError('out_axes={oa} inconsistent with axes(out)={a}'.format(oa=out_axes, a=axes(out)))

    out_axes = elementwise_axes(args_axes, out_axes)

    if out is None:
        out = AxesArray(axes=out_axes)

    # Reshape out for NumPy so that out broadcast axes are on the left

    rout = reaxe(out, axes_append(axes_sub(out_axes, args_axes), args_axes))

    # Reshape args so they are ordered consistently with output
    rargs = (reaxe(arg, args_axes, broadcast=True) for arg in args)

    return rargs, rout, out


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


def abs(x, out=None):
    (rx,), rout, out = elementwise_reaxe(args=(x,), out=out)
    np.abs(rx, out=rout)
    return out


def add(x, y, out=None):
    (rx, ry), rout, out = elementwise_reaxe(args=(x, y), out=out)
    np.add(rx, ry, out=rout)
    return out


def cos(x, out=None):
    (rx,), rout, out = elementwise_reaxe(args=(x,), out=out)
    np.cos(rx, out=rout)
    return out


def divide(x, y, out=None):
    (rx, ry), rout, out = elementwise_reaxe(args=(x, y), out=out)
    np.divde(rx, ry, out=rout)
    return out


def dot(x, y, reduction_axes=None, out_axes=None, out=None):
    if out is not None:
        if out_axes is None:
            out_axes = out.axes
        else:
            if out_axes != out.axes:
                raise ValueError('out_axes does not match out.axes')

    x_axes = axes(x)
    y_axes = axes(y)

    reduction_axes, s_axes, out_axes = dot_axes(x_axes, y_axes, reduction_axes=reduction_axes, out_axes=out_axes)

    if out is None:
        out = AxesArray(axes=out_axes)

    # Split reductions into groups of contiguous axes that can be done with one np.dot
    x_base_axes = axes(x._base())
    y_base_axes = axes(y._base())
    # reduction ordered by x with broadcasts at end
    x_reduction_axes = axes_intersect(x_base_axes, reduction_axes)
    y_reduction_axes = axes_intersect(y_base_axes, reduction_axes)

    # np.dot needs a single reduction axis
    def axis_groups(base_axes, reduction_axes):
        next_idx = -1
        groups = []
        idx_group = []
        for axis in reduction_axes:
            idx = base_axes.index(axis)
            if len(idx_group) == 0 or next_idx == idx:
                idx_group.append(axis)

            else:
                groups.append(idx_group)
                idx_group = [axis]
            if idx >= 0:
                next_idx = idx + 1

        if idx_group:
            groups.append(idx_group)

        return groups

    x_groups = axis_groups(x_base_axes, reduction_axes)
    y_groups = axis_groups(y_base_axes, reduction_axes)
    if len(x_groups) == 1 and len(y_groups) == 1 and x_groups[0] == y_groups[0]:
        reduce = x_groups[0]
        xr_axes = axes_sub(x_axes, reduction_axes)
        xr_axes.extend(reduce)
        xr = reaxe(x, xr_axes, broadcast=True)
        yr_axes = axes_sub(y_axes, reduce)
        if len(yr_axes) < 2:
            yr_axes.insert(0, reduce)
            yr = reaxe(y, yr_axes)
        else:
            yr_axes[-1:-1] = reduce
            yr = reaxe(y, yr_axes, broadcast=True)
        outr = reaxe(out, out_axes)
        if len(out_axes) == 0:
            outr[()] = np.dot(xr, yr)
        else:
            np.dot(xr, yr, outr)
        return out

    raise ValueError('Complex dot not supported')


def exp(x, out=None):
    (rx,), rout, out = elementwise_reaxe(args=(x,), out=out)
    np.exp(rx, out=rout)
    return out


def log(x, out=None):
    (rx,), rout, out = elementwise_reaxe(args=(x,), out=out)
    np.log(rx, out=rout)
    return out


def maximum(x, y, out=None):
    (rx, ry), rout, out = elementwise_reaxe(args=(x, y), out=out)
    np.maximum(rx, ry, out=rout)
    return out


def minimum(x, y, out=None):
    (rx, ry), rout, out = elementwise_reaxe(args=(x, y), out=out)
    np.minimum(rx, ry, out=rout)
    return out


def multiply(x, y, out=None):
    (rx, ry), rout, out = elementwise_reaxe(args=(x, y), out=out)
    np.multiply(rx, ry, out=rout)
    return out


def negative(x, out=None):
    (rx,), rout, out = elementwise_reaxe(args=(x,), out=out)
    np.negative(rx, out=rout)
    return out


def power(x, y, out=None):
    (rx, ry), rout, out = elementwise_reaxe(args=(x, y), out=out)
    np.power(rx, ry, out=rout)
    return out


def reciprocal(x, out=None):
    (rx,), rout, out = elementwise_reaxe(args=(x,), out=out)
    np.reciprocal(rx, out=rout)
    return out


def sign(x, out=None):
    (rx,), rout, out = elementwise_reaxe(args=(x,), out=out)
    np.sign(rx, out=rout)
    return out


def sin(x, out=None):
    (rx,), rout, out = elementwise_reaxe(args=(x,), out=out)
    np.sin(rx, out=rout)
    return out


def square(x, out=None):
    (rx,), rout, out = elementwise_reaxe(args=(x,), out=out)
    np.square(rx, out=rout)
    return out


def subtract(x, y, out=None):
    (rx, ry), rout, out = elementwise_reaxe(args=(x, y), out=out)
    np.subtract(rx, ry, out=rout)
    return out


def tanh(x, out=None):
    (rx,), rout, out = elementwise_reaxe(args=(x,), out=out)
    np.tanh(rx, out=rout)
    return out


class AxesArray(np.ndarray, HasAxes):
    def __new__(cls, axes, base=None, broadcast=False, **kargs):
        shape = tuple(axes_shape(axes))
        if base is not None:
            base_array = base._base()
            base_axes = base_array.axes
            base_strides = base_array.strides

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

            return super(AxesArray, cls).__new__(cls, buffer=base_array, shape=shape, strides=strides, **kargs)

        return super(AxesArray, cls).__new__(cls, shape=shape, **kargs)

    def __init__(self, axes, **kargs):
        super(AxesArray, self).__init__(**kargs)
        self.__axes = axes

    def _base(self):
        return self.base or self

    @property
    def axes(self):
        return self.__axes

    def reaxe(self, axes, broadcast=False):
        if axes == self.axes:
            return self

        return AxesArray(base=self, axes=axes, broadcast=broadcast)

    def __abs__(self):
        return abs(self)

    def __add__(self, other):
        if not isinstance(other, AxesArray):
            raise ValueError()

        return add(self, other)

    def __div__(self, other):
        if not isinstance(other, AxesArray):
            raise ValueError()

        return divide(self, other)

    def __mul__(self, other):
        if not isinstance(other, HasAxes):
            raise ValueError()

        return multiply(self, other)

    def __pow__(self, other):
        if not isinstance(other, AxesArray):
            raise ValueError()

        return power(self, other)

    def __radd__(self, other):
        if not isinstance(other, AxesArray):
            raise ValueError()

        return add(other, self)

    def __rdiv__(self, other):
        if not isinstance(other, AxesArray):
            raise ValueError()

        return divide(other, self)

    def __rmul__(self, other):
        if not isinstance(other, AxesArray):
            raise ValueError()

        return multiply(other, self)

    def __rpow__(self, other):
        if not isinstance(other, AxesArray):
            raise ValueError()

        return power(other, self)

    def __rsub__(self, other):
        if not isinstance(other, AxesArray):
            raise ValueError()

        return subtract(other, self)

    def __sub__(self, other):
        if not isinstance(other, AxesArray):
            raise ValueError()

        return subtract(self, other)


def reaxe_array(array, axes, new_axes):
    axis_pos = [-1]*len(new_axes)
    for i, axis in enumerate(new_axes):
        pos = axes.index(axis)
        axis_pos[i] = pos
    old_strides = array.strides
    old_shape = array.shape
    new_strides = [old_strides[pos] for pos in axis_pos]
    new_shape = [old_shape[pos] for pos in axis_pos]
    return np.ndarray(shape=new_shape, dtype=array.dtype, buffer=array, strides=new_strides)


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


H = Axis('H', 3)
W = Axis('W', 4)
C = Axis('C', 2)

for idx in axes_generator((H,W,C), C):
    for idx1 in axes_generator((H,W,C), H, idx):
        print idx1


a = AxesArray(axes=(H,W))
b = AxesArray(axes=(W,H))
c = AxesArray(axes=(W,C,H))
d = AxesArray(axes=(H,W,C))
for h in range(H.length):
    for w in range(W.length):
        a[h,w] = 10*h+w
        b[w,h] = 10*w+h

v = AxesArray(axes=(H,W))
w = AxesArray(axes=(H,))
v.fill(2)
w.fill(3)
v2 = dot(v, w)
print(v2)


print(a)
print(b)
add(a,b,out=c)
print(c)
print(c*2)
multiply(c,2,d)
e = d.reaxe((C, (H,W)))
print(e)


