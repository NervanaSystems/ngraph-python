import collections
import weakref
import inspect

from geon.backends.graph.names import NameableValue
from geon.backends.graph.environment import get_current_environment

# TODO This implementation only read the model description.  These model descriptions use
# TODO implicit allocation, with the exception of inputs and parameters.
# TODO Once the model description has been
# TODO read, we can analyze it and produce an intermediate graph in terms of primitive access
# TODO operations and tensor operations that compute on explicit allocations.


def get_all_defs():
    return Defmod.defs()


def find_all(tags=None, used_by=None, uses=None, types=None):
    """
    Find tensors satisfying criteria.

    :param tags: Required tags
    :param used_by: Restrict to tensors used by this tensor
    :param types: Restrict to these types, default Tensor.
    :return: a set of tensors.
    """
    result = set()
    visited = set()
    unvisited = set()
    if used_by is not None:
        if isinstance(used_by, collections.Iterable):
            unvisited.update(used_by)
        else:
            unvisited.add(used_by)
    else:
        unvisited.update([_ for _ in get_all_defs() if isinstance(_, Tensor)])

    tagset = set()
    if tags is collections.Iterable and not isinstance(tags, str):
        tagset.update(tags)
    elif tags is not None:
        tagset.add(tags)

    if types is None:
        types = (Tensor,)

    while unvisited:
        item = unvisited.pop()
        if item in visited:
            continue
        visited.add(item)
        if isinstance(item, types) and tagset.issubset(item.tags):
            result.add(item)
        unvisited.update(item.args)

    if uses is not None:
        usesset = set()
        unvisited = set()
        visited = set()

        if isinstance(uses, collections.Iterable):
            unvisited.update(uses)
        else:
            unvisited.add(uses)

        while unvisited:
            item = unvisited.pop()
            if item in visited:
                continue
            if item in result:
                usesset.add(item)
            unvisited.update(item.args)
        result.intersection_update(usesset)

    return result


class Defmod(NameableValue):
    """Base class for model definitions

    Handles generic printing and tracking object creation order.
    """

    def __init__(self, tags=None, **kargs):
        super(Defmod, self).__init__(**kargs)
        defs = Defmod.defs()
        self.seqid = len(defs)
        defs.append(self)

        self.tags = set()
        if tags is not None:
            if isinstance(tags, collections.Iterable) and not isinstance(tags, str):
                self.tags.update(tags)
            else:
                self.tags.add(tags)

        # TODO This is a good first cut for debugging info, but it would be nice to
        # TODO be able to reliably walk the stack back to user code rather than just
        # TODO back past this constructor
        frame = None
        try:
            frame = inspect.currentframe()
            while frame.f_locals.get('self', None) is self:
                frame = frame.f_back
            while frame:
                filename, lineno, function, code_context, index = inspect.getframeinfo(frame)
                if -1 == filename.find('geon/backends/graph'):
                    break
                frame = frame.f_back

            self.filename = filename
            self.lineno = lineno
            self.code_context = code_context
        finally:
            del frame

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
            name = '{' + self.name + '}'
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

            return '(' + result + ')'
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
                result = result + ', ' + s
            else:
                result = s
        return result

    def __repr__(self):
        return '{s}({body})'.format(s=self.__shortpr(), body=self._repr_body())


class Tensor(Defmod):
    """Any tensor-value"""

    def __init__(self, axes=None, batch_axes=None, dtype=None, args=None, **kargs):
        super(Tensor, self).__init__(**kargs)
        self._axes = Axes.as_axes(axes)
        self.batch_axes = Axes.as_axes(batch_axes)
        self.dtype = dtype

        if args is None:
            self.args = ()
        else:
            self.args = tuple(Tensor.as_tensor(_) for _ in args)
        for arg in self.args:
            arg.users.add(self)

        # Tensors that directly use the result
        self.users = weakref.WeakSet()  # Name assigned by user

    @property
    def size(self):
        return TensorSize(self)

    @property
    def axes(self):
        return self._axes

    def _repr_attrs(self, *attrs):
        return super(Tensor, self)._repr_attrs('_axes', 'batch_axes', 'dtype', 'args', *attrs)

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

    def __getitem__(self, key):
        return TensorGetItem(self, key)


class TensorSize(Tensor):
    def __init__(self, tensor, **kargs):
        super(TensorSize, self).__init__(axes=(), args=(tensor,))


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


class RecursiveTensor(Tensor):
    """A NumPy array with axes"""

    def __init__(self, axes, **kargs):
        super(RecursiveTensor, self).__init__(**kargs)

    def __setitem__(self, key, value):
        TensorSetItem(self, key, value)


class TensorSetItem(Defmod):
    def __init__(self, tensor, key, value, **kargs):
        super(TensorSetItem, self).__init__(**kargs)
        self.tensor = tensor
        self.key = key
        self.value = value

    def _repr_attrs(self, *attrs):
        return super(TensorSetItem, self)._repr_attrs('tensor', 'key', 'value', *attrs)


class TensorGetItem(Tensor):
    def __init__(self, tensor, key, **kargs):
        super(TensorGetItem, self).__init__(**kargs)
        self.tensor = tensor
        self.key = key

    def _repr_attrs(self, *attrs):
        return super(TensorGetItem, self)._repr_attrs('tensor', 'key', *attrs)


class VariableExpr(Defmod):
    def __init__(self, args, **kargs):
        super(VariableExpr, self).__init__(**kargs)
        self.args = args

    def _repr_attrs(self, *attrs):
        return super(VariableExpr, self)._repr_attrs('args', *attrs)

    def __add__(self, other):
        return VariableAdd(self, other)

    def __radd__(self, other):
        return VariableAdd(other, self)

    def __sub__(self, other):
        return VariableSub(self, other)

    def __rsub__(self, other):
        return VariableSub(other, self)


class VariableAdd(VariableExpr):
    def __init__(self, x, y, **kargs):
        super(VariableAdd, self).__init__(args=(x, y), **kargs)


class VariableSub(VariableExpr):
    def __init__(self, x, y, **kargs):
        super(VariableSub, self).__init__(args=(x, y), **kargs)


class Variable(VariableExpr):
    def __init__(self, kind, **kargs):
        super(Variable, self).__init__(args=(), **kargs)
        self.kind = kind

    def _repr_attrs(self, *attrs):
        return super(Variable, self)._repr_attrs('kind', *attrs)


class Parameter(Tensor):
    """A parameter to be trained"""

    def __init__(self, init=None, **kargs):
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


class ElementWise(ComputedTensor):
    """Element wise computation"""

    def __init__(self, e_axes=None, **kargs):
        super(ElementWise, self).__init__(**kargs)
        self.e_axes = Axes.as_axes(e_axes)

    def _repr_attrs(self, *attrs):
        return super(ElementWise, self)._repr_attrs('e_axes', *attrs)


class Reduction(ComputedTensor):
    def __init__(self, r_axes=None, **kargs):
        super(Reduction, self).__init__(**kargs)
        self.r_axes = Axes.as_axes(r_axes)

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


class decrement(ComputedTensor):
    def __init__(self, parameter, change, **kargs):
        super(decrement, self).__init__(args=(parameter, change), **kargs)


class divide(ElementWise):
    def __init__(self, x, y, **kargs):
        super(divide, self).__init__(args=(x, y), **kargs)


class doall(ComputedTensor):
    def __init__(self, all, **kargs):
        super(doall, self).__init__(args=all, **kargs)


class dot(ComputedTensor):
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


class softmax(Tensor):
    def __init__(self, x, **kargs):
        super(softmax, self).__init__(args=(x,), **kargs)


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
        self._length = length
        self.dependents = dependents
        self.like = like
        if self.like is not None:
            self.name = self.like.name

    def __getitem__(self, item):
        return AxisIdx(axis=self, idx=item)

    @property
    def length(self):
        return self._length or AxisLength(self)

    @length.setter
    def length(self, length):
        AxisSetLength(self, length)

    def _repr_attrs(self, *attrs):
        return super(Axis, self)._repr_attrs('_length', 'dependents', 'like', *attrs)


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
