from contextlib import contextmanager
import weakref
import numbers

import numpy as np

import neon.backends.backend

from geon.backends.graph.names import NameableValue
import geon.backends.graph.typing as typing
from geon.backends.graph.errors import *
from geon.backends.graph.environment import get_current_environment, get_current_ops
import geon.backends.graph.arrayaxes as arrayaxes
from geon.backends.graph.arrayaxes import AxesComp, ValueAxesComp, BatchAxes, AxesIntersectComp, AxesSubComp, AxesAppendComp, tensor_axes


class Op(NameableValue):
    """Any operation that can be in an AST"""

    def __init__(self, **kwds):
        self.predecessors = weakref.WeakSet()
        self._adjoints = None
        super(Op, self).__init__(**kwds)
        ops = get_current_ops()
        if ops is not None:
            ops.append(self)

    def parameters(self):
        """Return all parameters used in computing this node"""
        params = []
        visited = set()
        unvisited = [self]

        while unvisited:
            node = unvisited.pop()
            visited.add(node)
            if isinstance(node, Variable):
                params.append(node)
            unvisited.extend(node.inputs)

        return params

    @property
    def inputs(self):
       return ()

    @staticmethod
    def get_ordered_ops(op, ordered_ops, include_outs):
        """
        Get dependent ops ordered for autodiff.
        """
        if op not in ordered_ops:
            if isinstance(op, ComputationOp):
                for arg in op.inputs:
                    Op.get_ordered_ops(arg, ordered_ops, include_outs)
                output = op.output
                if include_outs and output is not None and op is not output:
                    Op.get_ordered_ops(op.output, ordered_ops, include_outs)
            ordered_ops.append(op)

    @property
    def adjoints(self):
        if self._adjoints is not None:
            return self._adjoints

        self._adjoints = weakref.WeakKeyDictionary()
        ordered_ops = []
        Op.get_ordered_ops(self, ordered_ops, False)
        self._adjoints[self] = ones(axes=tensor_axes(self))
        for o in reversed(ordered_ops):
            o.generate_adjoints(self._adjoints, self._adjoints[o], *o.inputs)
        return self._adjoints

    @staticmethod
    def ordered_ops(results, include_outs):
        ordered_ops = []
        for result in results:
            Op.get_ordered_ops(result, ordered_ops, include_outs)
        return ordered_ops

    @staticmethod
    def analyze_liveness(results, ordered_ops):
        liveness = [set() for _ in ordered_ops]
        i = len(liveness) - 1
        for result in results:
            liveness[i].add(result.output)
        while i > 0:
            op = ordered_ops[i]
            prealive = liveness[i - 1]
            alive = set(liveness[i])
            if isinstance(op, Tensor):
                output = op.output
                alive.discard(output)
                for arg in op.inputs:
                    alive.add(arg.output)
                prealive |= alive
            i = i - 1
        return liveness

    @staticmethod
    def as_op(x):
        if isinstance(x, Tensor):
            return x

        return Constant(x)

    @property
    def ops(self):
        return []

    def allocate(self, evaluator):
        pass

    def __str__(self):
        return '<{cl}:{id}>'.format(cl=self.__class__.__name__, id=id(self))


class Tensor(Op):

    def __init__(self, graph_type=None, **kwds):
        super(Tensor, self).__init__(**kwds)
        self.graph_type = graph_type

        # Ops that directly use the result
        self.users = weakref.WeakSet()  # Name assigned by user

    @property
    def output(self):
        return self

    @property
    def axes(self):
        return ValueAxesComp(self)

    def generate_add_delta(self, adjoints, delta):
        if self not in adjoints:
            adjoints[self] = delta
        else:
            adjoints[self] = delta + adjoints[self]

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

    def __setitem__(self, key, val):
        return SetItem(self, key, val)

    def __axes__(self):
        return self.axes

    # Required for parameter initializers
    @property
    def shape(self):
        return self.__axes__()


arrayaxes.ObjectWithAxes.register(Tensor)


class ComputationOp(Tensor):
    """
    An TensorOp is the result of some sort of operation.
    """
    def __init__(self, args, out=None, batch_axes=None, **kargs):
        super(ComputationOp, self).__init__(**kargs)
        self.__args = tuple(Op.as_op(arg) for arg in args)

        for arg in self.inputs:
            arg.users.add(self)

        self.batch_axes = AxesComp.as_axes(batch_axes or BatchAxes())

        if out is None:
            out = empty(axes=tensor_axes(self))
        self.output = out

    def add_dependencies(self):
        self.output.users.add(self)

    @property
    def inputs(self):
        return self.__args

    @property
    def output(self):
        return self.__out

    @output.setter
    def output(self, value):
        self.__out = value
        if value is not self:
            value.users.add(self)


class RNG(ComputationOp):
    def __init__(self, seed=None, **kargs):
        super(RNG, self).__init__(args=(), **kargs)
        self.seed = seed

    @property
    def axes(self):
        return AxesComp.as_axes(())

    def uniform(self, low=0.0, high=1.0, size=None, **kargs):
        return Uniform(rng=self,low=low, high=high, size=size, **kargs)

    def evaluate(self, evaluator, out):
        return evaluator.rng(seed=self.seed)


class RNGOp(ComputationOp):
    def __init__(self, rng, axes, **kargs):
        self.__axes = axes
        super(RNGOp, self).__init__(args=(rng,), **kargs)

    @property
    def axes(self):
        return self.__axes


class Uniform(RNGOp):
    def __init__(self, low=0.0, high=1.0, size=None, **kargs):
        super(Uniform, self).__init__(axes=size, **kargs)
        self.low = low
        self.high = high

    def evaluate(self, evaluator, out, rng):
        return evaluator.rng_uniform(rng, self.low, self.high, out)


class VoidOp(ComputationOp):
    def __init__(self, **kargs):
        super(VoidOp, self).__init__(**kargs)

    @property
    def axes(self):
        return tensor_axes(self.output)


class decrement(VoidOp):
    def __init__(self, parameter, change, **kargs):
        super(decrement, self).__init__(out=parameter, args=(parameter, change), **kargs)

    def evaluate(self, evaluator, out, parameter, change):
        evaluator.update(parameter, change)
        return out


class SetItem(VoidOp):
    def __init__(self, tensor, item, val, **kargs):
        super(SetItem, self).__init__(args=(tensor, val), out=tensor, **kargs)
        self.item = item

    def evaluate(self, evaluator, out, tensor, val):
        evaluator.set_item(tensor, self.item, val)
        pass


class doall(VoidOp):
    def __init__(self, all, **kargs):
        super(doall, self).__init__(args=all, out=all[-1].output, **kargs)

    def evaluate(self, evaluator, out, *args):
        return out


class ElementWise(ComputationOp):
    def __init__(self, **kargs):
        super(ElementWise, self).__init__(**kargs)

    @property
    def axes(self):
        inputs = self.inputs
        result = tensor_axes(self.inputs[0])
        for input in inputs[1:]:
            result = AxesAppendComp(result, tensor_axes(input))
        return result


class trace(ElementWise):
    def __init__(self, x, label=None, **kargs):
        super(trace, self).__init__(args=(x,), **kargs)
        self.label = label

    def evaluate(self, evaluator, out, x):
        evaluator.trace(x, self.label, out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, trace(delta, label='d'+self.label))


class AllocationOp(Tensor):
    def __init__(self, axes=None, dtype=None, **kargs):
        super(AllocationOp, self).__init__(graph_type=typing.Array[AxesComp.as_axes(axes), dtype], **kargs)
        self.aliases = weakref.WeakSet()

    @property
    def axes(self):
        return self.graph_type.axes


class Reaxe(AllocationOp):
    def __init__(self, axes, **kargs):
        super(AllocationOp, self).__init__(axes=axes, graph_type=typing.Array[AxesComp.as_axes(axes), dtype], **kargs)




class placeholder(AllocationOp):
    """
    Can be set externally.
    """
    def __init__(self, **kargs):
        super(placeholder, self).__init__(**kargs)
        self.__axes = ValueAxesComp(self)

    def evaluate(self, evaluator):
        return evaluator.input_value(self)

    def generate_adjoints(self, tape, delta):
        pass

    def __axes__(self):
        return self.__axes

    @property
    def value(self):
        return get_current_environment()[self]

    @value.setter
    def value(self, value):
        environment = get_current_environment()
        environment[self] = value
        # op = Op.as_op(value)
        # environment.set_cached_resolved_tensor_axes(self, tensor_axes(op))


class Constant(AllocationOp):
    """
    A constant that appears in a graph.
    """
    def __init__(self, const, **kargs):
        if isinstance(const, arrayaxes.ObjectWithAxes):
            # TODO: Figure out what to do for axes here
            super(Constant, self).__init__(axes=tensor_axes(const), dtype=np.dtype(type(const)), **kargs)
        else:
            super(Constant, self).__init__(axes=(), dtype=np.dtype(type(const)), **kargs)
        self.const = const

    def evaluate(self, evaluator):
        return evaluator.constant(self.const, axes=evaluator.get_resolved_tensor_axes(self), dtype=self.graph_type.dtype)

    def generate_adjoints(self, tape, delta):
        pass

    @property
    def axes(self):
        return AxesComp.as_axes((()))

    def __str__(self):
        return '<{cl} ({const})>'.format(cl=self.__class__.__name__, const=self.const)


class absolute(ElementWise):
    def __init__(self, x, **kargs):
        super(absolute, self).__init__(args=(x,), **kargs)

    def evaluate(self, evaluator, out, x):
        return evaluator.absolute(x, out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, sig(x)*delta)


class add(ElementWise):
    def __init__(self, x, y, **kargs):
        super(add, self).__init__(args=(x, y), **kargs)

    def evaluate(self, evaluator, out, x, y):
        return evaluator.add(x, y, out)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, sum(delta, out_axes=tensor_axes(x)))
        y.generate_add_delta(adjoints, sum(delta, out_axes=tensor_axes(y)))


class cos(ElementWise):
    def __init__(self, x, **kargs):
        super(cos, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*sin(x))

    def evaluate(self, evaluator, out, x):
        return evaluator.cos(x, out)


# This makes the derivative simpler if we need it
def divide(x, y, **kargs):
    result = multiply(x, reciprocal(y), **kargs)
    return result


class dot(ComputationOp):
    def __init__(self, x, y, reduction_axes=None, out_axes=None, **kargs):
        self.out_axes = AxesComp.as_axes(out_axes)
        if reduction_axes is None:
            self.reduction_axes = AxesIntersectComp(tensor_axes(x), tensor_axes(y))
        else:
            self.reduction_axes = AxesComp.as_axes(reduction_axes)

        if out_axes is not None:
            self.reduction_axes = AxesSubComp(self.reduction_axes, self.out_axes)

        super(dot, self).__init__(args=(x, y), **kargs)

    def evaluate(self, evaluator, out, x, y):
        resolved_reduction_axes = evaluator.get_resolved_axes(self.reduction_axes)
        return evaluator.dot(x, y, resolved_reduction_axes, out)

    @property
    def axes(self):
        if self.out_axes:
            return self.out_axes
        else:
            x, y = self.inputs
            x_axes = tensor_axes(x)
            y_axes = tensor_axes(y)
            return AxesAppendComp(AxesSubComp(x_axes, self.reduction_axes), AxesSubComp(y_axes, self.reduction_axes))

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, dot(delta, y, out_axes=tensor_axes(x)))
        y.generate_add_delta(adjoints, dot(x, delta, out_axes=tensor_axes(y)))


class softmax(ComputationOp):
    def __init__(self, x, **kargs):
        super(softmax, self).__init__(args=(x,), **kargs)

    def evaluate(self, evaluator, out, x):
        return evaluator.softmax(x, evaluator.get_resolved_axes(self.batch_axes), out)

    @property
    def axes(self):
        x, = self.inputs
        return tensor_axes(x)

    def generate_adjoints(self, adjoints, delta, x):
        z = delta*self
        zs = sum(z, reduction_axes=AxesSubComp(tensor_axes(x), self.batch_axes))
        x.generate_add_delta(adjoints, (z-zs*self))


class sum(ComputationOp):
    def __init__(self, x, reduction_axes=None, out_axes=None, **kargs):
        self.out_axes = AxesComp.as_axes(out_axes)
        if reduction_axes is None:
            if out_axes is None:
                self.reduction_axes = tensor_axes(x)
            else:
                self.reduction_axes = AxesSubComp(tensor_axes(x), self.out_axes)
        else:
            self.reduction_axes = AxesComp.as_axes(reduction_axes)
        super(sum, self).__init__(args=(x,), **kargs)

    def evaluate(self, evaluator, out, x):
        resolved_reduction_axes = evaluator.get_resolved_axes(self.reduction_axes)
        return evaluator.sum(x, resolved_reduction_axes, out)

    @property
    def axes(self):
        if self.out_axes is not None:
            return self.out_axes
        return AxesSubComp(tensor_axes(self.inputs[0]), self.reduction_axes)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta)


class empty(AllocationOp):
    def __init__(self, **kargs):
        super(empty, self).__init__(**kargs)

    def generate_adjoints(self, adjoints, delta):
        pass

    def allocate(self, evaluator):
        return evaluator.empty(axes=evaluator.get_resolved_tensor_axes(self), dtype=self.graph_type.dtype)

    def evaluate(self, evaluator):
        return evaluator.value(self)


class Slice(ComputationOp):
    def __init__(self, slices, x, **kargs):
        super(Slice, self).__init__(args=(x,), **kargs)
        self.slices = slices

    def evaluate(self, evaluator, out, x):
        # TODO This is like an allocate,
        pass


class Pad(ComputationOp):
    def __init__(self, axes, slice, x, **kargs):
        super(Pad, self).__init__(args=(x,), **kargs)
        self._axes = axes
        self.slice = slice

    @property
    def axes(self):
        return self._axes

    def evaluate(self, evaluator, out, x):
        return evaluator.pad(x, self.slice, out)

    def generate_adjoints(self, adjoints, delta, x):
        pass


class Variable(AllocationOp):
    def __init__(self, init, **kargs):
        super(Variable, self).__init__(**kargs)
        self.init = init

    def generate_adjoints(self, adjoints, delta):
        pass

    def evaluate(self, evaluator):
        return evaluator.value(self)

    def allocate(self, evaluator):
        return evaluator.empty(axes=evaluator.get_resolved_tensor_axes(self), dtype=self.graph_type.dtype)

    def initializer(self, evaluator, value):
        if self.init:
            self.init.fill(self)

    @property
    def value(self):
        return get_current_environment()[self]


class exp(ElementWise):
    def __init__(self, x, **kargs):
        super(exp, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta)

    def evaluate(self, evaluator, out, x):
        return evaluator.exp(x, out)


class log(ElementWise):
    def __init__(self, x, **kargs):
        super(log, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta/x)

    def evaluate(self, evaluator, out, x):
        return evaluator.log(x, out)


class safelog(log):
    def evaluate(self, evaluator, out, x):
        return evaluator.safelog(x, out)


class maximum(ElementWise):
    def __init__(self, x, y, **kargs):
        super(maximum, self).__init__(args=(x, y), **kargs)

    def evaluate(self, evaluator, out, x, y):
        return evaluator.maximum(x, y, out=out)

    def generate_adjoints(self, adjoints, delta, x, y):
        p, n = posneg(x-y)
        x.generate_add_delta(delta*p)
        y.generate_add_delta(delta*n)


class minimum(ElementWise):
    def __init__(self, x, y, **kargs):
        super(minimum, self).__init__(args=(x, y), **kargs)

    def evaluate(self, evaluator, out, x, y):
        return evaluator.minimum(x, y, out=out)

    def generate_adjoints(self, adjoints, delta, x, y):
        p, n = posneg(y-x)
        x.generate_add_delta(delta*p)
        y.generate_add_delta(delta*n)


class multiply(ElementWise):
    def __init__(self, x, y, **kargs):
        super(multiply, self).__init__(args=(x, y), **kargs)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, sum(delta*y, out_axes=tensor_axes(x)))
        y.generate_add_delta(adjoints, sum(x*delta, out_axes=tensor_axes(y)))


    def evaluate(self, evaluator, out, x, y):
        return evaluator.multiply(x, y, out)


class negative(ElementWise):
    def __init__(self, x, **kargs):
        super(negative, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -delta)

    def evaluate(self, evaluator, out, x):
        return evaluator.negative(x, out)


class ones(AllocationOp):
    def __init__(self, **kargs):
        super(ones, self).__init__(**kargs)

    def generate_adjoints(self, adjoints, delta):
        pass

    def evaluate(self, evaluator):
        return evaluator.ones(axes=evaluator.get_resolved_tensor_axes(self), dtype=self.graph_type.dtype)


class reciprocal(ElementWise):
    def __init__(self, x, **kargs):
        super(reciprocal, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -self*self*delta)

    def evaluate(self, evaluator, out, x):
        return evaluator.reciprocal(x, out)


class sgn(ElementWise):
    def __init__(self, x, **kargs):
        super(sgn, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        # Zero
        pass

    def evaluate(self, evaluator, out, x):
        return evaluator.sign(x, out)


class sig(ElementWise):
    """Sigmoid"""
    def __init__(self, x, **kargs):
        super(sig, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*self*(1.0-self))

    def evaluate(self, evaluator, out, x):
        return evaluator.sig(x, out)

class sin(ElementWise):
    def __init__(self, x, **kargs):
        super(sin, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*cos(x))

    def evaluate(self, evaluator, out, x):
        return evaluator.sin(x, out)


class sqrt(ElementWise):
    def __init__(self, x, **kargs):
        super(sqrt, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, .5*delta*self)

    def evaluate(self, evaluator, out, x):
        return evaluator.sqrt(x, out)


class square(ElementWise):
    def __init__(self, x, **kargs):
        super(square, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, 2.0*delta*x)

    def evaluate(self, evaluator, out, x):
        return evaluator.square(x, out)


class subtract(ElementWise):
    def __init__(self, x, y, **kargs):
        super(subtract, self).__init__(args=(x, y), **kargs)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, -delta)

    def evaluate(self, evaluator, out, x, y):
        return evaluator.subtract(x, y, out)


class tanh(ElementWise):
    def __init__(self, x, **kargs):
        super(tanh, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*(1.0-self*self))

    def evaluate(self, evaluator, out, x):
        return evaluator.tanh(x, out)


class zeros(AllocationOp):
    def __init__(self, **kargs):
        super(zeros, self).__init__(**kargs)

    def generate_adjoints(self, adjoints, delta):
        pass

    def evaluate(self, evaluator):
        return evaluator.zeros(axes=evaluator.get_resolved_tensor_axes(self), dtype=self.graph_type.dtype)


def deriv(dep, indep):
    return dep.adjoints[indep]


def cross_entropy_multi(y, t):
    return -sum(safelog(y) * t)*(1.0/np.log(2))


def cross_entropy_binary(y, t):
    a = - safelog(y) * t
    b = - safelog(1 - y) * (1 - t)
    return sum(a + b)
