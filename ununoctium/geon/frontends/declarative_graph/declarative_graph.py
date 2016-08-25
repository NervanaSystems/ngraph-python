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
from builtins import str
import collections
# import weakref
# import inspect

# from geon.backends.graph.names import NameableValue
from geon.op_graph.nodes import Node


# TODO This implementation only read the model description.  These model descriptions use
# TODO implicit allocation, with the exception of inputs and parameters.
# TODO Once the model description has been
# TODO read, we can analyze it and produce an intermediate graph in terms of primitive access
# TODO operations and tensor operations that compute on explicit allocations.


def get_all_defs():
    """TODO."""
    return Defmod.defs()


def find_all(tags=None, used_by=None, uses=None, types=None):
    """
    Find tensors satisfying criteria.

    Arguments:
      tags: Required tags
      used_by: Restrict to tensors used by this tensor
      uses: TODO
      types: Restrict to these types, default Tensor.

    Returns:
        A set of tensors.
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


class Defmod(Node):
    """
    Base class for model definitions.

    Handles generic printing and tracking object creation order.
    """

    def __init__(self, **kwargs):
        super(Defmod, self).__init__(**kwargs)
        defs = Defmod.defs()
        self.seqid = len(defs)
        defs.append(self)

    _defs = []

    @staticmethod
    def defs():
        """TODO."""
        return Defmod._defs

    def as_node(self, x):
        """
        TODO.

        Arguments:
          x: TODO

        Returns:

        """
        return Tensor.as_tensor(x)


class Tensor(Defmod):
    """Any tensor-value"""

    def __init__(self, axes=None, batch_axes=None, dtype=None, **kwargs):
        super(Tensor, self).__init__(**kwargs)
        self._axes = Axes.as_axes(axes)
        self.batch_axes = Axes.as_axes(batch_axes)
        self.dtype = dtype

    @property
    def size(self):
        """TODO."""
        return TensorSize(self)

    @property
    def axes(self):
        """TODO."""
        return self._axes

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(
            Tensor,
            self)._repr_attrs(
            '_axes',
            'batch_axes',
            'dtype',
            'args',
            *attrs)

    @staticmethod
    def as_tensor(tensor):
        """
        TODO.

        Arguments:
          tensor: TODO

        Returns:

        """
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

    def __truediv__(self, val):
        return divide(self, val)

    def __rdiv__(self, val):
        return divide(val, self)

    def __pow__(self, val):
        return Pow(self, val)

    def __rpow__(self, val):
        return Pow(val, self)

    def __getitem__(self, key):
        return TensorGetItem(self, key)


class Pow(Tensor):
    """TODO."""
    pass


class TensorSize(Tensor):
    """TODO."""

    def __init__(self, tensor, **kwargs):
        super(TensorSize, self).__init__(axes=(), args=(tensor,))


class LiteralTensor(Tensor):
    """A literal value in the definition."""

    def __init__(self, value, **kwargs):
        super(LiteralTensor, self).__init__(**kwargs)
        self.value = value

    @property
    def graph_label(self):
        """TODO."""
        return str(self.value)

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(LiteralTensor, self)._repr_attrs('value', *attrs)


class ArrayWithAxes(Tensor):
    """A NumPy array with axes"""

    def __init__(self, array, axes, **kwargs):
        super(ArrayWithAxes, self).__init__(
            axes=axes, dtype=array.dtype, **kwargs)
        self.array = array

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(ArrayWithAxes, self)._repr_attrs('array', *attrs)


class RecursiveTensor(Tensor):
    """A NumPy array with axes"""

    def __init__(self, axes, **kwargs):
        super(RecursiveTensor, self).__init__(**kwargs)

    def __setitem__(self, key, value):
        TensorSetItem(self, key, value)


class TensorSetItem(Defmod):
    """TODO."""

    def __init__(self, tensor, key, value, **kwargs):
        super(TensorSetItem, self).__init__(**kwargs)
        self.tensor = tensor
        self.key = key
        self.value = value

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(
            TensorSetItem,
            self)._repr_attrs(
            'tensor',
            'key',
            'value',
            *attrs)


class TensorGetItem(Tensor):
    """TODO."""

    def __init__(self, tensor, key, **kwargs):
        super(TensorGetItem, self).__init__(**kwargs)
        self.tensor = tensor
        self.key = key

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(TensorGetItem, self)._repr_attrs('tensor', 'key', *attrs)


class VarExpr(Defmod):
    """TODO."""

    def __init__(self, args, **kwargs):
        super(VarExpr, self).__init__(**kwargs)
        self.args = args

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(VarExpr, self)._repr_attrs('args', *attrs)

    def __add__(self, other):
        return VarAdd(self, other)

    def __radd__(self, other):
        return VarAdd(other, self)

    def __sub__(self, other):
        return VarSub(self, other)

    def __rsub__(self, other):
        return VarSub(other, self)


class VarAdd(VarExpr):
    """TODO."""

    def __init__(self, x, y, **kwargs):
        super(VarAdd, self).__init__(args=(x, y), **kwargs)


class VarSub(VarExpr):
    """TODO."""

    def __init__(self, x, y, **kwargs):
        super(VarSub, self).__init__(args=(x, y), **kwargs)


class Var(VarExpr):
    """TODO."""

    def __init__(self, kind, **kwargs):
        super(Var, self).__init__(args=(), **kwargs)
        self.kind = kind

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(Var, self)._repr_attrs('kind', *attrs)


class Variable(Tensor):
    """A parameter to be trained"""

    def __init__(self, init=None, **kwargs):
        super(Variable, self).__init__(**kwargs)
        self.init = init

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(Variable, self)._repr_attrs('init', *attrs)


class input(Tensor):
    """A value that will be supplied."""

    def __init__(self, axes, **kwargs):
        super(input, self).__init__(axes=axes, **kwargs)


class ComputedTensor(Tensor):
    """An computation that produces a tensor value"""

    def __init__(self, **kwargs):
        super(ComputedTensor, self).__init__(**kwargs)

    @property
    def graph_label(self):
        """TODO."""
        return type(self).__name__

    @property
    def axes(self):
        """TODO."""
        return ValueAxes(self)


class ElementWise(ComputedTensor):
    """Element wise computation"""

    def __init__(self, e_axes=None, **kwargs):
        super(ElementWise, self).__init__(**kwargs)
        self.e_axes = Axes.as_axes(e_axes)

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(ElementWise, self)._repr_attrs('e_axes', *attrs)


class Reduction(ComputedTensor):
    """TODO."""

    def __init__(self, r_axes=None, **kwargs):
        super(Reduction, self).__init__(**kwargs)
        self.r_axes = Axes.as_axes(r_axes)

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(Reduction, self)._repr_attrs('r_axes', *attrs)


class absolute(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(absolute, self).__init__(args=(x,), **kwargs)


class add(ElementWise):
    """TODO."""

    def __init__(self, x, y, **kwargs):
        super(add, self).__init__(args=(x, y), **kwargs)


class cos(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(cos, self).__init__(args=(x,), **kwargs)


class decrement(ComputedTensor):
    """TODO."""

    def __init__(self, parameter, change, **kwargs):
        super(decrement, self).__init__(args=(parameter, change), **kwargs)


class divide(ElementWise):
    """TODO."""

    def __init__(self, x, y, **kwargs):
        super(divide, self).__init__(args=(x, y), **kwargs)


class doall(ComputedTensor):
    """TODO."""

    def __init__(self, all, **kwargs):
        super(doall, self).__init__(args=all, **kwargs)


class dot(ComputedTensor):
    """TODO."""

    def __init__(self, x, y, reduction_axes=None, **kwargs):
        super(dot, self).__init__(args=(x, y), **kwargs)
        self.r_axes = Axes.as_axes(reduction_axes)

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(dot, self)._repr_attrs('r_axes', *attrs)


class exp(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(exp, self).__init__(args=(x,), **kwargs)


class log(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(log, self).__init__(args=(x,), **kwargs)


class maximum(ElementWise):
    """TODO."""

    def __init__(self, x, y, **kwargs):
        super(maximum, self).__init__(args=(x, y), **kwargs)


class minimum(ElementWise):
    """TODO."""

    def __init__(self, x, y, **kwargs):
        super(minimum, self).__init__(args=(x, y), **kwargs)


class multiply(ElementWise):
    """TODO."""

    def __init__(self, x, y, **kwargs):
        super(multiply, self).__init__(args=(x, y), **kwargs)


class negative(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(negative, self).__init__(args=(x,), **kwargs)


class reciprocal(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(reciprocal, self).__init__(args=(x,), **kwargs)


class sign(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(sign, self).__init__(args=(x,), **kwargs)


class sigmoid(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(sigmoid, self).__init__(args=(x,), **kwargs)


class softmax(ComputedTensor):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(softmax, self).__init__(args=(x,), **kwargs)


class sin(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(sin, self).__init__(args=(x,), **kwargs)


class sqrt(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(sqrt, self).__init__(args=(x,), **kwargs)


class square(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(square, self).__init__(args=(x,), **kwargs)


class subtract(ElementWise):
    """TODO."""

    def __init__(self, x, y, **kwargs):
        super(subtract, self).__init__(args=(x, y), **kwargs)


class sum(Reduction):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(sum, self).__init__(args=(x,), **kwargs)


class tanh(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(tanh, self).__init__(args=(x,), **kwargs)


class deriv(Tensor):
    """TODO."""

    def __init__(self, dep, indep, **kwargs):
        super(deriv, self).__init__(args=(dep, indep), **kwargs)


class Axis(Defmod):
    """TODO."""

    def __init__(self, length=None, dependents=None, like=None, **kwargs):
        super(Axis, self).__init__(**kwargs)
        self._length = length
        self.dependents = dependents
        self.like = like
        if self.like is not None:
            self.name = self.like.name

    def __getitem__(self, item):
        return AxisIdx(axis=self, idx=item)

    @property
    def length(self):
        """TODO."""
        return self._length or AxisLength(self)

    @length.setter
    def length(self, length):
        """
        TODO.

        Arguments:
          length: TODO

        Returns:

        """
        AxisSetLength(self, length)

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(
            Axis,
            self)._repr_attrs(
            '_length',
            'dependents',
            'like',
            *attrs)


class AxisLength(Defmod):
    """TODO."""

    def __init__(self, axis, **kwargs):
        super(AxisLength, self).__init__(**kwargs)
        self.axis = axis

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(AxisLength, self)._repr_attrs('axis', *attrs)


class AxisSetLength(Defmod):
    """TODO."""

    def __init__(self, axis, length, **kwargs):
        super(AxisSetLength, self).__init__(**kwargs)
        self.axis = axis
        self.length = length

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(AxisSetLength, self)._repr_attrs('axis', 'length', *attrs)


class AxisIdx(Defmod):
    """TODO."""

    def __init__(self, axis, idx, **kwargs):
        super(AxisIdx, self).__init__(**kwargs)
        self.axis = axis
        self.idx = idx

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(AxisIdx, self)._repr_attrs('axis', 'idx')


class AxisVariable(Defmod):
    """TODO."""

    def __init__(self, **kwargs):
        super(AxisVariable, self).__init__(**kwargs)


class Axes(Defmod):
    """TODO."""

    def __init__(self, **kwargs):
        super(Axes, self).__init__(**kwargs)

    @staticmethod
    def as_axes(axes):
        """
        TODO.

        Arguments:
          axes: TODO

        Returns:

        """
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
    """TODO."""

    def __init__(self, axes, **kwargs):
        super(LiteralAxes, self).__init__(**kwargs)
        if not isinstance(axes, collections.Iterable):
            axes = (axes,)
        self.axes = axes

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(LiteralAxes, self)._repr_attrs('axes', *attrs)


class ValueAxes(Axes):
    """TODO."""

    def __init__(self, value, **kwargs):
        super(ValueAxes, self).__init__(**kwargs)
        self.value = value

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(ValueAxes, self)._repr_attrs('value', *attrs)


class CombineAxes(Axes):
    """TODO."""

    def __init__(self, args, **kwargs):
        super(CombineAxes, self).__init__(**kwargs)
        self.args = [Axes.as_axes(_) for _ in args]

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return super(CombineAxes, self)._repr_attrs('args', *attrs)


class AppendAxes(CombineAxes):
    """TODO."""

    def __init__(self, *args, **kwargs):
        super(AppendAxes, self).__init__(args=args, **kwargs)


class SubAxes(CombineAxes):
    """TODO."""

    def __init__(self, *args, **kwargs):
        super(SubAxes, self).__init__(args=args, **kwargs)


class IntersectAxes(CombineAxes):
    """TODO."""

    def __init__(self, *args, **kwargs):
        super(IntersectAxes, self).__init__(args=args, **kwargs)
