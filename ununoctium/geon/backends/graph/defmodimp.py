import collections
import weakref

from geon.backends.graph.names import NameableValue
from geon.backends.graph.environment import get_current_environment


def get_all_defs():
    return Defmod.defs()


class Defmod(NameableValue):
    """Base class for model definitions

    Handles generic printing and tracking object creation order.
    """
    def __init__(self, **kargs):
        super(Defmod, self).__init__(**kargs)
        defs = Defmod.defs()
        self.seqid = len(defs)
        defs.append(self)

    @staticmethod
    def defs():
        try:
            defs = get_current_environment()[Defmod]
        except KeyError:
            defs = []
            get_current_environment()[Defmod] = defs
        return defs

    def _repr_body(self):
        return self._abbrev_args(self._repr_attrs())

    def _repr_attrs(self, *attrs):
        return attrs

    def __shortpr(self):
        name = ''
        if self.name is not None:
            name = '{'+self.name+'}'
        return '{seqid}:{cls}{name}'.format(name=name, seqid=self.seqid, cls=self.__class__.__name__)

    def _abbrev_value(self, value):
        if isinstance(value, Defmod):
            return value.__shortpr()
        elif isinstance(value, tuple):
            result = ''
            for _ in value:
                s = self._abbrev_value(_)
                if result:
                    result = result + ', ' + s
                else:
                    result = s

            return '('+result+')'
        else:
            return '{v}'.format(v=value)

    def _abbrev_args(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        result = ''
        for key in keys:
            val = self.__getattribute__(key)
            if val is None:
                continue
            s = '{key}={val}'.format(key=key, val=self._abbrev_value(val))
            if result:
                result = result+', '+s
            else:
                result = s
        return result

    def __repr__(self):
        return '{s}({body})'.format(s=self.__shortpr(), body=self._repr_body())


class Tensor(Defmod):
    """Any tensor-value"""
    def __init__(self, axes=None, dtype=None, **kargs):
        super(Tensor, self).__init__(**kargs)
        self._axes = Axes.as_axes(axes)
        self.dtype = dtype

        # Tensors that directly use the result
        self.users = weakref.WeakSet()  # Name assigned by user

    @property
    def axes(self):
        return self._axes

    @property
    def args(self):
        return ()

    def parameters(self):
        """Return all parameters used in computing this node"""
        params = []
        visited = set()
        unvisited = [self]

        while unvisited:
            node = unvisited.pop()
            visited.add(node)
            if isinstance(node, Parameter):
                params.append(node)
            unvisited.extend(node.args)

        return params

    def _repr_attrs(self, *attrs):
        return super(Tensor, self)._repr_attrs('axes', 'dtype', *attrs)

    @staticmethod
    def as_tensor(tensor):
        if tensor is None or isinstance(tensor, Tensor):
            return tensor
        return LiteralTensor(tensor)

    # Magic methods for builtin operations we want to use for creating nodes
    def __neg__(self):
        return negative(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return absolute(self)

    def __add__(self, val):
        return add(self, val)

    def __radd__(self, val):
        return add(val, self)

    def __sub__(self, val):
        return subtract(self, val)

    def __rsub__(self, val):
        return subtract(val, self)

    def __mul__(self, val):
        return multiply(self, val)

    def __rmul__(self, val):
        return multiply(val, self)

    def __div__(self, val):
        return divide(self, val)

    def __rdiv__(self, val):
        return divide(val, self)

    def __pow__(self, val):
        return Pow(self, val)

    def __rpow__(self, val):
        return Pow(val, self)


class LiteralTensor(Tensor):
    """A literal value in the definition."""
    def __init__(self, value, **kargs):
        super(LiteralTensor, self).__init__(**kargs)
        self.value = value

    def _repr_attrs(self, *attrs):
        return super(LiteralTensor, self)._repr_attrs('value', *attrs)


class ArrayWithAxes(Tensor):
    """A NumPy array with axes"""
    def __init__(self, array, axes, **kargs):
        super(ArrayWithAxes, self).__init__(axes=axes, dtype=array.dtype, **kargs)
        self.array = array

    def _repr_attrs(self, *attrs):
        return super(ArrayWithAxes, self)._repr_attrs('array', *attrs)


class Parameter(Tensor):
    """A parameter to be trained"""
    def __init__(self, init, **kargs):
        super(Parameter, self).__init__(**kargs)
        self.init = init

    def _repr_attrs(self, *attrs):
        return super(Parameter, self)._repr_attrs('init', *attrs)


class input(Tensor):
    """A value that will be supplied."""
    def __init__(self, axes, **kargs):
        super(input, self).__init__(axes=axes, **kargs)


class ComputedTensor(Tensor):
    """An computation that produces a tensor value"""
    def __init__(self, **kargs):
        super(ComputedTensor, self).__init__(**kargs)

    @property
    def axes(self):
        return ValueAxes(self)


class Args(Defmod):
    """Something that has tensor arguments"""
    def __init__(self, args, **kargs):
        super(Args, self).__init__(**kargs)
        self._args = tuple(Tensor.as_tensor(_) for _ in args)
        for arg in self._args:
            arg.users.add(self)

    @property
    def args(self):
        return self._args

    def _repr_attrs(self, *attrs):
        return super(Args, self)._repr_attrs('args', *attrs)


class ElementWise(Args, ComputedTensor):
    """Element wise computation"""
    def __init__(self, e_axes=None, **kargs):
        super(ElementWise, self).__init__(**kargs)
        self.e_axes = Axes.as_axes(e_axes)

    def _repr_attrs(self, *attrs):
        return super(ElementWise, self)._repr_attrs('e_axes', *attrs)


class Reduction(Args, ComputedTensor):
    def __init__(self, r_axes=None, **kargs):
        super(Reduction, self).__init__(**kargs)
        self.r_aces = Axes.as_axes(r_axes)

    def _repr_attrs(self, *attrs):
        return super(Reduction, self)._repr_attrs('r_axes', *attrs)


class absolute(ElementWise):
    def __init__(self, x, **kargs):
        super(absolute, self).__init__(args=(x,), **kargs)


class add(ElementWise):
    def __init__(self, x, y, **kargs):
        super(add, self).__init__(args=(x, y), **kargs)


class cos(ElementWise):
    def __init__(self, x, **kargs):
        super(cos, self).__init__(args=(x,), **kargs)


class decrement(Args):
    def __init__(self, parameter, change, **kargs):
        super(decrement, self).__init__(args=(parameter, change), **kargs)


class divide(ElementWise):
    def __init__(self, x, y, **kargs):
        super(divide, self).__init__(args=(x, y), **kargs)


class doall(Args):
    def __init__(self, all, **kargs):
        super(doall, self).__init__(args=all, **kargs)


class dot(Args, ComputedTensor):
    def __init__(self, x, y, reduction_axes=None, **kargs):
        super(dot, self).__init__(args=(x, y), **kargs)
        self.r_axes = Axes.as_axes(reduction_axes)

    def _repr_attrs(self, *attrs):
        return super(dot, self)._repr_attrs('r_axes', *attrs)


class sum(Reduction):
    def __init__(self, x, **kargs):
        super(sum, self).__init__(args=(x,), **kargs)

class exp(ElementWise):
    def __init__(self, x, **kargs):
        super(exp, self).__init__(args=(x,), **kargs)


class log(ElementWise):
    def __init__(self, x, **kargs):
        super(log, self).__init__(args=(x,), **kargs)


class maximum(ElementWise):
    def __init__(self, x, y, **kargs):
        super(maximum, self).__init__(args=(x, y), **kargs)


class minimum(ElementWise):
    def __init__(self, x, y, **kargs):
        super(minimum, self).__init__(args=(x, y), **kargs)


class multiply(ElementWise):
    def __init__(self, x, y, **kargs):
        super(multiply, self).__init__(args=(x, y), **kargs)


class negative(ElementWise):
    def __init__(self, x, **kargs):
        super(negative, self).__init__(args=(x,), **kargs)


class ones(Tensor):
    def __init__(self, **kargs):
        super(ones, self).__init__(**kargs)


class reciprocal(ElementWise):
    def __init__(self, x, **kargs):
        super(reciprocal, self).__init__(args=(x,), **kargs)


class sgn(ElementWise):
    def __init__(self, x, **kargs):
        super(sgn, self).__init__(args=(x,), **kargs)


class sig(ElementWise):
    """Sigmoid"""
    def __init__(self, x, **kargs):
        super(sig, self).__init__(args=(x,), **kargs)


class sin(ElementWise):
    def __init__(self, x, **kargs):
        super(sin, self).__init__(args=(x,), **kargs)


class sqrt(ElementWise):
    def __init__(self, x, **kargs):
        super(sqrt, self).__init__(args=(x,), **kargs)


class square(ElementWise):
    def __init__(self, x, **kargs):
        super(square, self).__init__(args=(x,), **kargs)


class subtract(ElementWise):
    def __init__(self, x, y, **kargs):
        super(subtract, self).__init__(args=(x, y), **kargs)


class sum(Reduction):
    def __init__(self, x, **kargs):
        super(sum, self).__init__(args=(x,), **kargs)


class tanh(ElementWise):
    def __init__(self, x, **kargs):
        super(tanh, self).__init__(args=(x,), **kargs)


class zeros(Tensor):
    def __init__(self, **kargs):
        super(zeros, self).__init__(**kargs)


class deriv(Tensor):
    def __init__(self, dep, indep, **kargs):
        super(deriv, self).__init__(args=(dep, indep), **kargs)


class Axis(Defmod):
    def __init__(self, length=None, dependents=None, like=None, **kargs):
        super(Axis, self).__init__(**kargs)
        self.__length = length
        self.dependents = dependents
        self.like = like
        if self.like is not None:
            self.name = self.like.name

    def __getitem__(self, item):
        return AxisIdx(axis=self, idx=item)

    @property
    def length(self):
        return self.__length or AxisLength(self)

    @length.setter
    def length(self, length):
        return AxisSetLength(self, length)

    def _repr_attrs(self, *attrs):
        return super(Axis, self)._repr_attrs('length', 'dependents', 'like', *attrs)


class AxisLength(Defmod):
    def __init__(self, axis, **kargs):
        super(AxisLength, self).__init__(**kargs)
        self.axis = axis

    def _repr_attrs(self, *attrs):
        return super(AxisLength, self)._repr_attrs('axis', *attrs)


class AxisSetLength(Defmod):
    def __init__(self, axis, length, **kargs):
        super(AxisSetLength, self).__init__(**kargs)
        self.axis = axis
        self.length = length

    def _repr_attrs(self, *attrs):
        return super(AxisSetLength, self)._repr_attrs('axis', 'length', *attrs)


class AxisIdx(Defmod):
    def __init__(self, axis, idx, **kargs):
        super(AxisIdx, self).__init__(**kargs)
        self.axis = axis
        self.idx = idx

    def _repr_attrs(self, *attrs):
        return super(AxisIdx, self)._repr_attrs('axis', 'idx')


class AxisVariable(Defmod):
    def __init__(self, **kargs):
        super(AxisVariable, self).__init__(**kargs)


class Axes(Defmod):
    def __init__(self, **kargs):
        super(Axes, self).__init__(**kargs)

    @staticmethod
    def as_axes(axes):
        if axes is None or isinstance(axes, Axes):
            return axes
        return LiteralAxes(axes)

    def __add__(self, axes):
        return AppendAxes(self, axes)

    def __radd__(self, axes):
        return AppendAxes(axes, self)

    def __sub__(self, axes):
        return SubAxes(self, axes)

    def __rsub__(self, axes):
        return SubAxes(axes, self)

    def __and__(self, other):
        return IntersectAxes(self, other)

    def __rand__(self, other):
        return IntersectAxes(other, self)


class LiteralAxes(Axes):
    def __init__(self, axes, **kargs):
        super(LiteralAxes, self).__init__(**kargs)
        if not isinstance(axes, collections.Iterable):
            axes = (axes,)
        self.axes = axes

    def _repr_attrs(self, *attrs):
        return super(LiteralAxes, self)._repr_attrs('axes', *attrs)


class ValueAxes(Axes):
    def __init__(self, value, **kargs):
        super(ValueAxes, self).__init__(**kargs)
        self.value = value

    def _repr_attrs(self, *attrs):
        return super(ValueAxes, self)._repr_attrs('value', *attrs)


class CombineAxes(Axes):
    def __init__(self, args, **kargs):
        super(CombineAxes, self).__init__(**kargs)
        self.args = [Axes.as_axes(_) for _ in args]

    def _repr_attrs(self, *attrs):
        return super(CombineAxes, self)._repr_attrs('args', *attrs)


class AppendAxes(CombineAxes):
    def __init__(self, *args, **kargs):
        super(AppendAxes, self).__init__(args=args, **kargs)


class SubAxes(CombineAxes):
    def __init__(self, *args, **kargs):
        super(SubAxes, self).__init__(args=args, **kargs)


class IntersectAxes(CombineAxes):
    def __init__(self, *args, **kargs):
        super(IntersectAxes, self).__init__(args=args, **kargs)

