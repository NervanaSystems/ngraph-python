import abc
import numbers
import collections
import numpy as np

import geon.backends.graph.arrayaxes as arrayaxes
from geon.backends.graph.arrayaxes import Axis, axes_sub, axes_intersect, axes_append, axes_contains
from geon.backends.graph.arrayaxes import elementwise_axes, axes, reaxe, dot_axes


# This is a partial implementation of axes on top of NumPy

# TODO See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html for information
# TODO about properly subclassing ndarray.

# TODO Track the computation so that we can interactively autodiff

# TODO Only the simpler axis cases are handled

# TODO AxisIDs are mostly missing

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
        xr = arrayaxes.reaxe(x, xr_axes, broadcast=True)
        yr_axes = axes_sub(y_axes, reduce)
        if len(yr_axes) < 2:
            yr_axes.insert(0, reduce)
            yr = arrayaxes.reaxe(y, yr_axes)
        else:
            yr_axes[-1:-1] = reduce
            yr = arrayaxes.reaxe(y, yr_axes, broadcast=True)
        outr = arrayaxes.reaxe(out, out_axes)
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
        shape = tuple(arrayaxes.axes_shape(axes))
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
                component_axes = arrayaxes.flatten_axes(axis)
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
        if not isinstance(other, AxesArray):
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


H = Axis('H', 3)
W = Axis('W', 4)
C = Axis('C', 2)

for idx in arrayaxes.axes_generator((H,W,C), C):
    for idx1 in arrayaxes.axes_generator((H,W,C), H, idx):
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


