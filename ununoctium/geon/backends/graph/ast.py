from contextlib import contextmanager
import weakref
import numbers
import collections

import numpy as np

import neon.initializers.initializer as initializer

from geon.backends.graph.names import NameableValue
from geon.backends.graph.nodes import Node
import geon.backends.graph.typing as typing
from geon.backends.graph.errors import *
from geon.backends.graph.environment import get_current_environment, get_current_ops
import geon.backends.graph.arrayaxes as arrayaxes
from geon.backends.graph.arrayaxes import tensor_axes, get_batch_axes, set_batch_axes, find_axes_in_axes

from mpi4py import MPI

comm = MPI.COMM_WORLD

class Op(NameableValue, Node):
    """Any operation that can be in an AST"""

    def __init__(self, **kwds):
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

    @staticmethod
    def get_ordered_ops(op, ordered_ops):
        """
        Get dependent ops ordered for autodiff.
        """
        Node.visit_input_closure([op], lambda o: ordered_ops.append(o))

    @property
    def adjoints(self):
        if self._adjoints is not None:
            return self._adjoints

        self._adjoints = weakref.WeakKeyDictionary()
        ordered_ops = []
        Op.get_ordered_ops(self, ordered_ops)
        self._adjoints[self] = ones(axes=tensor_sample_axes(self))
        for o in reversed(ordered_ops):
            if o in self._adjoints:
                scale = o.scale
                adjoint = self._adjoints[o]
                if scale != 1.0:
                    adjoint = adjoint * scale
                o.generate_adjoints(self._adjoints, adjoint, *o.inputs)
        return self._adjoints

    @staticmethod
    def ordered_ops(results):
        ordered_ops = []
        Node.visit_input_closure(results, lambda o: ordered_ops.append(o))
        return ordered_ops

    @staticmethod
    def analyze_liveness(results, ordered_ops):
        liveness = [set() for _ in ordered_ops]
        i = len(liveness) - 1
        for result in results:
            liveness[i].add(result)
        while i > 0:
            op = ordered_ops[i]
            prealive = liveness[i - 1]
            alive = set(liveness[i])
            if isinstance(op, Tensor):
                alive.discard(op)
                for arg in op.inputs:
                    alive.add(arg)
                prealive |= alive
            i = i - 1
        return liveness

    def as_node(self, x):
        return Op.as_op(x)

    @staticmethod
    def as_op(x):
        if isinstance(x, Tensor):
            return x

        return Constant(x)

    @property
    def ops(self):
        return []

    def evaluate(self, evaluator, *args):
        """Process op"""
        pass

    def sync(self, evaluator):
        """Make sure evaluator has local changes"""
        pass

    def __str__(self):
        return '<{cl}:{id}>'.format(cl=self.__class__.__name__, id=id(self))


class TensorAxesInfo(object):
    """Information about a use of a tensor with axes"""
    def __init__(self, axes, alloc=None, init=None, read_only=False, tags=(), dtype=np.float32, **kargs):
        super(TensorAxesInfo, self).__init__(**kargs)
        axes = tuple(axes)
        self.axes = axes
        self.views = weakref.WeakValueDictionary()
        self.alloc = alloc
        self.init = init
        self.read_only = read_only
        self.dtype = np.dtype(dtype)
        self.tags = set(tags)
        self.__tensor_description = None

    @property
    def tensor_description(self):
        if self.__tensor_description is None:
            self.__tensor_description = arrayaxes.TensorDescription(axes=self.axes, dtype=self.dtype)
        return self.__tensor_description

    @property
    def value(self):
        return self.tensor_description.value

    def set_tensor(self, evaluator, tensor):
        self.tensor_description.value = tensor
        for view in self.views.values():
            if view.tensor_description is self.tensor_description:
                continue
            view.update_tensor(evaluator)

    def generate_initializations(self, tensor):
        if self.init:
            self.init.fill(tensor)

    def allocate(self, evaluator):
        if self.tensor_description.value is None:
            if self.alloc is not None:
                tensor = self.alloc(evaluator, self.tensor_description)
            else:
                tensor = evaluator.empty(self.tensor_description)
            self.set_tensor(evaluator, tensor)
            self.tensor_description.value = tensor

    def get_or_default(self, axes, default_function):
        axes = arrayaxes.canonicalize_axes(axes)
        if self.views.has_key(axes):
            return self.views[axes]
        result = default_function()
        self.views[axes] = result
        return result

    def reaxe(self, reaxe):
        return self.get_or_default(reaxe, lambda : TensorReaxeViewInfo(tensor_axes_info=self, reaxes=reaxe, idx=len(self.views)))


class TensorViewInfo(object):
    """The use of a view of a tensor with axes"""
    def __init__(self, tensor_axes_info, idx, **kargs):
        super(TensorViewInfo, self).__init__(**kargs)
        self.tensor_axes_info = tensor_axes_info
        self.idx = idx

    def allocate(self, evaluator):
        if self.tensor_description.value is None:
            tensor = evaluator.empty(self.tensor_description)
            self.tensor_description.value = tensor

    @property
    def value(self):
        return self.tensor_description.value

    def update_tensor(self, evaluator):
        tensor_description = self.tensor_description
        tensor_description.value = evaluator.tensor_view(tensor_description)


class TensorReaxeViewInfo(TensorViewInfo):
    """The use of a reaxe view of a tensor with axes"""
    def __init__(self, reaxes, **kargs):
        super(TensorReaxeViewInfo, self).__init__(**kargs)
        self.reaxes = reaxes
        self.__tensor_description = None

    @property
    def tensor_description(self):
        if self.__tensor_description is None:
            self.__tensor_description = self.tensor_axes_info.tensor_description.reaxe(self.reaxes)
        return self.__tensor_description


class AxesComp(object):
    """A Computation for computing axes"""
    def __init__(self, axes=None, **kargs):
        super(AxesComp, self).__init__(**kargs)
        self.__axes__ = axes

    @staticmethod
    def as_axes(axes, **kargs):
        if isinstance(axes, AxesComp):
            return axes
        elif axes is None:
            return None
        else:
            return LiteralAxesComp(axes=axes, **kargs)

    @property
    def value(self):
        if self.__axes__ is None:
            self.__axes__ = self.resolve()
        return self.__axes__

    def resolve(self):
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


def sample_axes(x, **kargs):
    return AxesSubComp(AxesComp.as_axes(x, **kargs), get_batch_axes())


def tensor_sample_axes(x, **kargs):
    return sample_axes(tensor_axes(x), **kargs)


def tensor_batch_axes(x, **kargs):
    return batch_axes(tensor_axes(x), **kargs)


def batch_axes(x, **kargs):
    return AxesIntersectComp(AxesComp.as_axes(x, **kargs), get_batch_axes())


# This one should also work, but there are some bugs in axes/dot
def linear_map_axesa(in_axes, out_axes):
    return AxesSubComp(AxesAppendComp(in_axes, out_axes),
                       AxesIntersectComp(in_axes, out_axes))


def linear_map_axes(in_axes, out_axes):
    return AxesSubComp(AxesAppendComp(out_axes, in_axes),
                       AxesIntersectComp(in_axes, out_axes))


class LiteralAxesComp(AxesComp):
    """Actual axes are provided"""
    def __init__(self, **kargs):
        super(LiteralAxesComp, self).__init__(**kargs)


class ValueAxesComp(AxesComp):
    """Determine axes from value computed by x"""
    def __init__(self, x, **kargs):
        super(ValueAxesComp, self).__init__(**kargs)
        self.x = x

    def resolve(self):
        return self.x.resolved_axes


class AxesSubComp(AxesComp):
    """Result will be removal of axes in y from those in x"""
    def __init__(self, x, y, **kargs):
        super(AxesSubComp, self).__init__(**kargs)
        self.x = AxesComp.as_axes(x)
        self.y = AxesComp.as_axes(y)

    def resolve(self):
        x_axes = self.x.value
        y_axes = self.y.value
        return arrayaxes.axes_sub(x_axes, y_axes)


class AxesIntersectComp(AxesComp):
    def __init__(self, x, y, **kargs):
        super(AxesIntersectComp, self).__init__(**kargs)
        self.x = AxesComp.as_axes(x)
        self.y = AxesComp.as_axes(y)

    def resolve(self):
        x_axes = self.x.value
        y_axes = self.y.value
        return arrayaxes.axes_intersect(x_axes, y_axes)


class AxesAppendComp(AxesComp):
    def __init__(self, x, y, **kargs):
        super(AxesAppendComp, self).__init__(**kargs)
        self.x = AxesComp.as_axes(x)
        self.y = AxesComp.as_axes(y)

    def resolve(self):
        x_axes = self.x.value
        y_axes = self.y.value
        return arrayaxes.axes_append(x_axes, y_axes)


class Tensor(Op):

    def __init__(self, graph_type=None, scale=1, **kwds):
        super(Tensor, self).__init__(**kwds)
        self.graph_type = graph_type
        self.dtype = None
        if self.graph_type is not None:
            self.dtype = self.graph_type.dtype
        self.__tensor_axes_info = None
        self.__call_info = None

        # Derivative will be scaled by this if not 1.0
        self.scale = scale

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
        return power(self, val)

    def __rpow__(self, val):
        return power(val, self)

    # Python uses eq for comparing keys
    #def __eq__(self, val):
    #    return equal(self, val)

    #def __ne__(self, val):
    #    return not_equal(self, val)

    def __lt__(self, val):
        return less(self, val)

    def __gt__(self, val):
        return greater(self, val)

    def __le__(self, val):
        return less_equal(self, val)

    def __ge__(self, val):
        return greater_equal(self, val)

    def __setitem__(self, key, val):
        return SetItem(self, key, val)

    def __axes__(self):
        return self.axes

    @property
    def value(self):
        return self.tensor_axes_info.tensor_description.value

    @property
    def tensor_axes_info(self):
        if self.__tensor_axes_info is None:
            self.__tensor_axes_info = self.compute_tensor_axes_info()
        return self.__tensor_axes_info

    def compute_tensor_axes_info(self):
        dtype = np.float32
        if self.dtype is not None:
            dtype = self.dtype
        return TensorAxesInfo(self.axes.value, dtype=dtype)

    @property
    def call_info(self):
        if self.__call_info is None:
            self.__call_info = self.compute_call_info()
        return self.__call_info

    def compute_call_info(self):
        return [self.reaxe(self.resolved_axes)]

    def evaluate_call_info(self, evaluator, *args):
        call_args = [arg.value for arg in args]
        self.evaluate(evaluator, *call_args)

    @property
    def resolved_axes(self):
        return self.tensor_axes_info.axes

    def reaxe(self, reaxe):
        return self.tensor_axes_info.reaxe(reaxe)

    # Required for parameter initializers
    @property
    def shape(self):
        return self.__axes__()

    def mean(self, **kargs):
        return mean(self, **kargs)


arrayaxes.ObjectWithAxes.register(Tensor)


class ComputationOp(Tensor):
    """
    An TensorOp is the result of some sort of operation.
    """
    def __init__(self, out=None, dtype=np.float32, batch_axes=None, **kargs):
        super(ComputationOp, self).__init__(**kargs)
        self.dtype = dtype

        for arg in self.inputs:
            arg.users.add(self)

        if batch_axes is None:
            batch_axes = get_batch_axes()

        self.batch_axes = AxesComp.as_axes(batch_axes)


class RNG(ComputationOp):
    def __init__(self, seed=None, **kargs):
        super(RNG, self).__init__(args=(), **kargs)
        self.seed = seed

    def compute_tensor_axes_info(self):
        tensor_axes_info = super(RNG, self).compute_tensor_axes_info()
        tensor_axes_info.alloc = lambda evaluator, tensor_description : evaluator.rng(seed=self.seed)
        return tensor_axes_info

    @property
    def axes(self):
        return AxesComp.as_axes(())

    def uniform(self, low=0.0, high=1.0, size=None, **kargs):
        return Uniform(rng=self,low=low, high=high, size=size, **kargs)

    def allocate(self, evaluator):
        return evaluator.rng(seed=self.seed)


class RNGOp(ComputationOp):
    def __init__(self, rng, axes, **kargs):
        self.__axes = axes
        super(RNGOp, self).__init__(args=(rng,), **kargs)

    @property
    def axes(self):
        return self.__axes

    def compute_call_info(self):
        rng, = self.inputs
        call_info = super(RNGOp, self).compute_call_info()
        call_info.append(rng.reaxe(rng.resolved_axes))
        return call_info


class Uniform(RNGOp):
    def __init__(self, low=0.0, high=1.0, size=None, **kargs):
        super(Uniform, self).__init__(axes=size, **kargs)
        self.low = low
        self.high = high

    def evaluate(self, evaluator, out, rng):
        evaluator.rng_uniform(rng, self.low, self.high, out)


class VoidOp(ComputationOp):
    def __init__(self, **kargs):
        super(VoidOp, self).__init__(**kargs)
        self.__axes = AxesComp.as_axes(())

    @property
    def axes(self):
        return self.__axes

    def compute_call_info(self):
        # No out
        return []


class decrement(VoidOp):
    def __init__(self, parameter, delta, **kargs):
        super(decrement, self).__init__(out=parameter, args=(parameter, delta), **kargs)

    def compute_call_info(self):
        parameter, delta = self.inputs
        return [parameter.reaxe(parameter.resolved_axes), delta.reaxe(parameter.resolved_axes)]

    def evaluate(self, evaluator, parameter, change):
        evaluator.update(parameter, change)


class SetItem(VoidOp):
    def __init__(self, tensor, item, val, **kargs):
        super(SetItem, self).__init__(args=(tensor, val), out=tensor, **kargs)
        self.item = item

    def compute_call_info(self):
        tensor, val = self.inputs
        call_info = super(SetItem, self).compute_call_info()
        call_info.append(tensor.reaxe(tensor.resolved_axes))
        call_info.append(val.reaxe(tensor.resolved_axes))
        return call_info

    def evaluate(self, evaluator, tensor, val):
        evaluator.set_item(tensor, self.item, val)


class doall(VoidOp):
    def __init__(self, all, **kargs):
        super(doall, self).__init__(args=all, out=all[-1], **kargs)


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

    def compute_call_info(self):
        ci = super(ElementWise, self).compute_call_info()
        for arg in self.inputs:
            ci.append(arg.reaxe(self.resolved_axes))
        return ci


class AllReduce(ElementWise):
    def __init__(self, x, **kargs):
        super(AllReduce, self).__init__(args=(x,), **kargs)

    def evaluate(self, evaluator, out, x):
        x_val = x # read data from GPU to CPU -- expensive!
        recv_buffer = np.zeros(shape=x.shape, dtype=x.dtype)
        comm.Allreduce(x_val, recv_buffer, op= MPI.SUM)
        recv_buffer = recv_buffer / comm.Get_size() # Normalize the results to the number of MPI threads    
        out[:] = recv_buffer


class trace(ElementWise):
    def __init__(self, x, label=None, **kargs):
        super(trace, self).__init__(args=(x,), **kargs)
        self.label = label

    def evaluate(self, evaluator, out, x):
        evaluator.trace(x, self.label, out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, trace(delta, label='d'+self.label))


class AllocationOp(Tensor):
    def __init__(self, axes=None, init=None, dtype=np.float32, tags=(), **kargs):
        super(AllocationOp, self).__init__(graph_type=typing.Array[AxesComp.as_axes(axes), dtype], **kargs)
        self.tensor_axes_info.init = init
        self.tensor_axes_info.tags.update(tags)

    @property
    def axes(self):
        return self.graph_type.axes


class placeholder(AllocationOp):
    """
    Can be set externally.
    """
    def __init__(self, **kargs):
        super(placeholder, self).__init__(**kargs)
        self.__axes = ValueAxesComp(self)

    def __axes__(self):
        return self.__axes

    def generate_adjoints(self, tape, delta):
        pass

    #TODO Find a better way to set parameters
    @property
    def value(self):
        return get_current_environment()[self]

    @value.setter
    def value(self, value):
        get_current_environment()[self] = value

    def sync(self, evaluator):
        value = self.value
        if isinstance(value, arrayaxes.Scalar):
            evaluator.fill(self.tensor_axes_info.tensor_description.value, value)
        else:
            evaluator.set_value(self, value)


class ConstantInit(VoidOp):
    def __init__(self, tensor, const, **kargs):
        super(ConstantInit, self).__init__(args=(tensor,), **kargs)
        self.const = const

    def compute_call_info(self):
        tensor, = self.inputs
        call_info = super(ConstantInit, self).compute_call_info()
        call_info.append(tensor.reaxe(tensor.resolved_axes))
        return call_info

    def evaluate(self, evaluator, tensor):
        evaluator.fill(tensor, self.const)


class Constant(AllocationOp):
    """
    A constant that appears in a graph.
    """
    def __init__(self, const, **kargs):
        super(Constant, self).__init__(axes=(), dtype=np.dtype(np.float32), init=self, **kargs)
        self.const = const

    def fill(self, c):
        ConstantInit(c, self.const)

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
        evaluator.absolute(x, out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, sig(x)*delta)


class add(ElementWise):
    def __init__(self, x, y, **kargs):
        super(add, self).__init__(args=(x, y), **kargs)

    def evaluate(self, evaluator, out, x, y):
        evaluator.add(x, y, out)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, sum(delta, out_axes=tensor_axes(x)))
        y.generate_add_delta(adjoints, sum(delta, out_axes=tensor_axes(y)))


class argmax(ComputationOp):
    def __init__(self, x, max_axes=None, **kargs):
        if max_axes is None:
            max_axes = tensor_sample_axes(x)
        self.max_axes = AxesComp.as_axes(max_axes)
        super(argmax, self).__init__(args=(x,), dtype=np.int64, **kargs)

    def compute_call_info(self):
        x, = self.inputs
        return [self.reaxe([self.axes.value]), x.reaxe([self.max_axes.value, self.axes.value])]

    def evaluate(self, evaluator, out, x):
        evaluator.argmax(x, out)

    @property
    def axes(self):
        return AxesSubComp(tensor_axes(self.inputs[0]), self.max_axes)


class argmin(ComputationOp):
    def __init__(self, x, min_axes=None, **kargs):
        if min_axes is None:
            min_axes = tensor_sample_axes
        self.min_axes = AxesComp.as_axes(min_axes)
        super(argmin, self).__init__(args=(x,), dtype=np.int64, **kargs)

    def compute_call_info(self):
        x, = self.inputs
        return [self.reaxe([self.axes.value]), x.reaxe([self.min_axes.value, self.axes.value])]

    def evaluate(self, evaluator, out, x):
        evaluator.argmin(x, out)

    @property
    def axes(self):
        return AxesSubComp(tensor_axes(self.inputs[0]), self.min_axes)


class cos(ElementWise):
    def __init__(self, x, **kargs):
        super(cos, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*sin(x))

    def evaluate(self, evaluator, out, x):
        evaluator.cos(x, out)


class divide(ElementWise):
    def __init__(self, x, y, **kargs):
        super(divide, self).__init__(args=(x, y), **kargs)

    def evaluate(self, evaluator, out, x, y):
        evaluator.divide(x, y, out)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta*self/x)
        y.generate_add_delta(adjoints, -delta*self/y)


# This makes the derivative simpler if we need it
def dividex(x, y, **kargs):
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

        self.multiply = False

        super(dot, self).__init__(args=(x, y), **kargs)

    def compute_call_info(self):
        x, y = self.inputs

        x_axes = x.axes.value
        y_axes = y.axes.value
        out_axes = self.axes.value
        red_axes = self.reduction_axes.value

        if len(x_axes) is 0 or len(y_axes) is 0:
            # TODO turn this into multiply ahead of time
            self.multiply = True
            return [self.reaxe(self.resolved_axes), x.reaxe(x.resolved_axes), y.reaxe(y.resolved_axes)]
            np.multiply(x, y, out=out)
            return

        xi = find_axes_in_axes(red_axes, x_axes)
        if xi == -1:
            raise IncompatibleShapesError()
        yi = find_axes_in_axes(red_axes, y_axes)
        if yi == -1:
            raise IncompatibleShapesError()

        xl = x_axes[0:xi]
        xr = x_axes[xi+len(red_axes):]
        yl = y_axes[0:yi]
        yr = y_axes[yi+len(red_axes):]

        al = arrayaxes.axes_append(xl, xr)
        br = arrayaxes.axes_append(yl, yr)

        a = x.reaxe((al, red_axes))
        b = y.reaxe((red_axes, br))
        if arrayaxes.axes_intersect(al,br):
            # Can't handle yet
            raise IncompatibleShapesError()
        o = self.reaxe((al, br))
        return [o, a, b]

    def evaluate(self, evaluator, out, x, y):
        if self.multiply:
            evaluator.multiply(x, y, out)
        else:
            evaluator.dot(x, y, out)

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


class ElementWiseBoolean(ElementWise):
    def __init__(self, x, y, dtype=np.dtype(bool), **kargs):
        super(ElementWiseBoolean, self).__init__(args=(x, y), dtype=dtype, **kargs)


class equal(ElementWiseBoolean):
    def evaluate(self, evaluator, out, x, y):
        evaluator.equal(x, y, out)


class not_equal(ElementWiseBoolean):
    def evaluate(self, evaluator, out, x, y):
        evaluator.not_equal(x, y, out)


class greater(ElementWiseBoolean):
    def evaluate(self, evaluator, out, x, y):
        evaluator.greater(x, y, out)


class less(ElementWiseBoolean):
    def evaluate(self, evaluator, out, x, y):
        evaluator.less(x, y, out)


class greater_equal(ElementWiseBoolean):
    def evaluate(self, evaluator, out, x, y):
        evaluator.greater_equal(x, y, out)


class less_equal(ElementWiseBoolean):
    def evaluate(self, evaluator, out, x, y):
        evaluator.less_equal(x, y, out)


class softmax(ComputationOp):
    def __init__(self, x, **kargs):
        m = Temporary(axes=AxesComp.as_axes(get_batch_axes()))
        super(softmax, self).__init__(args=(x, m), **kargs)

    def compute_call_info(self):
        x, m = self.inputs
        batch_axes = self.batch_axes.value
        softmax_axes = arrayaxes.axes_sub(x.resolved_axes, batch_axes)
        if softmax_axes == ():
            raise ValueError('Empty softmax')

        xs = x.reaxe([softmax_axes, batch_axes])
        ms = m.reaxe([softmax_axes, batch_axes])
        out = self.reaxe([softmax_axes, batch_axes])
        return [out, xs, m.reaxe(batch_axes), ms]

    def evaluate(self, evaluator, out, x, m, ms):
        evaluator.max(x, 0, m)
        evaluator.subtract(x, ms, out)
        evaluator.exp(out, out)
        evaluator.sum(out, 0, m)
        evaluator.divide(out, ms, out)

    @property
    def axes(self):
        x, m = self.inputs
        return tensor_axes(x)

    def generate_adjoints(self, adjoints, delta, x, m):
        z = delta*self
        zs = sum(z, reduction_axes=AxesSubComp(tensor_axes(x), self.batch_axes))
        x.generate_add_delta(adjoints, (z-zs*self))


class sum(ComputationOp):
    def __init__(self, x, reduction_axes=None, out_axes=None, **kargs):
        self.out_axes = AxesComp.as_axes(out_axes)
        if reduction_axes is None:
            if out_axes is None:
                self.reduction_axes = sample_axes(x.axes)
            else:
                self.reduction_axes = AxesSubComp(x.axes, self.out_axes)
        else:
            self.reduction_axes = AxesComp.as_axes(reduction_axes)
        super(sum, self).__init__(args=(x,), **kargs)
        self.mode = None

    def compute_call_info(self):
        x, = self.inputs
        reduction_axes = self.reduction_axes.value

        if len(reduction_axes) == 0:
            # TODO do this as a reaxe to 1d or something
            xr = x.reaxe(self.resolved_axes)
            self.mode = 'copy'
            return [self.reaxe(self.resolved_axes), xr]
        else:
            x_axes = x.resolved_axes
            np_out_axes = self.resolved_axes
            sum_axes = [reduction_axes]
            sum_axes.extend(np_out_axes)
            self.mode = 0
            return [self.reaxe(np_out_axes), x.reaxe(sum_axes)]

    def evaluate(self, evaluator, out, x):
        if self.mode is 'copy':
            evaluator.copy(x, out)
        else:
            evaluator.sum(x, self.mode, out)

    @property
    def axes(self):
        if self.out_axes is not None:
            return self.out_axes
        return AxesSubComp(tensor_axes(self.inputs[0]), self.reduction_axes)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta)


class tensor_size(ComputationOp):
    def __init__(self, x, reduction_axes=None, **kargs):
        if reduction_axes is None:
            self.reduction_axes = tensor_axes(x)
        else:
            self.reduction_axes = AxesComp.as_axes(reduction_axes)
        super(tensor_size, self).__init__(args=(x,), **kargs)

    def evaluate(self, evaluator, out):
        resolved_reduction_axes = self.reduction_axes.value
        size = arrayaxes.axes_size(resolved_reduction_axes)
        evaluator.constant(size, out)

    @property
    def axes(self):
        return AxesComp.as_axes(())


class Slice(ComputationOp):
    def __init__(self, slices, x, **kargs):
        super(Slice, self).__init__(args=(x,), **kargs)
        self.slices = slices


class Pad(ComputationOp):
    def __init__(self, axes, slice, x, **kargs):
        super(Pad, self).__init__(args=(x,), **kargs)
        self._axes = axes
        self.slice = slice

    @property
    def axes(self):
        return self._axes

    def evaluate(self, evaluator, out, x):
        evaluator.pad(x, self.slice, out)

    def generate_adjoints(self, adjoints, delta, x):
        pass


class Variable(AllocationOp):
    def __init__(self, **kargs):
        super(Variable, self).__init__(**kargs)

    def generate_adjoints(self, adjoints, delta):
        pass

    def allocate(self, evaluator, tensor_allocation_info):
        self.empty(tensor_allocation_info)


class Temporary(AllocationOp):
    def __init__(self, **kargs):
        super(Temporary, self).__init__(tags=['temp'], **kargs)

    def generate_adjoints(self, adjoints, delta):
        pass

    def allocate(self, evaluator, tensor_allocation_info):
        self.empty(tensor_allocation_info)


class exp(ElementWise):
    def __init__(self, x, **kargs):
        super(exp, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta)

    def evaluate(self, evaluator, out, x):
        evaluator.exp(x, out)


class log(ElementWise):
    def __init__(self, x, **kargs):
        super(log, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta/x)

    def evaluate(self, evaluator, out, x):
        evaluator.log(x, out)


class safelog(log):
    def evaluate(self, evaluator, out, x):
        evaluator.safelog(x, out)


class maximum(ElementWise):
    def __init__(self, x, y, **kargs):
        super(maximum, self).__init__(args=(x, y), **kargs)

    def evaluate(self, evaluator, out, x, y):
        evaluator.maximum(x, y, out=out)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta*(self == x))
        y.generate_add_delta(adjoints, delta*(self == y))


class minimum(ElementWise):
    def __init__(self, x, y, **kargs):
        super(minimum, self).__init__(args=(x, y), **kargs)

    def evaluate(self, evaluator, out, x, y):
        evaluator.minimum(x, y, out=out)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta*(self == x))
        y.generate_add_delta(adjoints, delta*(self == y))


class multiply(ElementWise):
    def __init__(self, x, y, **kargs):
        super(multiply, self).__init__(args=(x, y), **kargs)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, sum(delta*y, out_axes=tensor_axes(x)))
        y.generate_add_delta(adjoints, sum(x*delta, out_axes=tensor_axes(y)))


    def evaluate(self, evaluator, out, x, y):
        evaluator.multiply(x, y, out)


class negative(ElementWise):
    def __init__(self, x, **kargs):
        super(negative, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -delta)

    def evaluate(self, evaluator, out, x):
        evaluator.negative(x, out)


class ones(AllocationOp):
    def __init__(self, **kargs):
        super(ones, self).__init__(**kargs)

    def generate_adjoints(self, adjoints, delta):
        pass

    def compute_tensor_axes_info(self):
        tensor_axes_info = super(ones, self).compute_tensor_axes_info()
        tensor_axes_info.alloc = self.allocate
        return tensor_axes_info

    def allocate(self, evaluator, tensor_allocation_info):
        return evaluator.ones(tensor_allocation_info)


class power(ElementWise):
    def __init__(self, x, y, **kargs):
        super(power, self).__init__(args=(x,), **kargs)

    def evaluate(self, evaluator, out, x, y):
        evaluator.pow(x, y, out)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta*y*self/x)
        y.generate_add_delta(adjoints, delta*self*log(x))


class reciprocal(ElementWise):
    def __init__(self, x, **kargs):
        super(reciprocal, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -self*self*delta)

    def evaluate(self, evaluator, out, x):
        evaluator.reciprocal(x, out)


class sgn(ElementWise):
    def __init__(self, x, **kargs):
        super(sgn, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        # Zero
        pass

    def evaluate(self, evaluator, out, x):
        evaluator.sign(x, out)


class sig(ElementWise):
    """Sigmoid"""
    def __init__(self, x, **kargs):
        super(sig, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*self*(1.0-self))

    def evaluate(self, evaluator, out, x):
        evaluator.sig(x, out)

class sin(ElementWise):
    def __init__(self, x, **kargs):
        super(sin, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*cos(x))

    def evaluate(self, evaluator, out, x):
        evaluator.sin(x, out)


class sqrt(ElementWise):
    def __init__(self, x, **kargs):
        super(sqrt, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, .5*delta*self)

    def evaluate(self, evaluator, out, x):
        evaluator.sqrt(x, out)


class square(ElementWise):
    def __init__(self, x, **kargs):
        super(square, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, 2.0*delta*x)

    def evaluate(self, evaluator, out, x):
        evaluator.square(x, out)


class subtract(ElementWise):
    def __init__(self, x, y, **kargs):
        super(subtract, self).__init__(args=(x, y), **kargs)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, -delta)

    def evaluate(self, evaluator, out, x, y):
        evaluator.subtract(x, y, out)


class tanh(ElementWise):
    def __init__(self, x, **kargs):
        super(tanh, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*(1.0-self*self))

    def evaluate(self, evaluator, out, x):
        evaluator.tanh(x, out)


class zeros(AllocationOp):
    def __init__(self, **kargs):
        super(zeros, self).__init__(**kargs)

    def generate_adjoints(self, adjoints, delta):
        pass

    def compute_tensor_axes_info(self):
        tensor_axes_info = super(ones, self).compute_tensor_axes_info()
        tensor_axes_info.alloc = self.allocate
        return tensor_axes_info

    def allocate(self, evaluator, tensor_allocation_info):
        return evaluator.zeros(tensor_allocation_info)


def mean(x, **kargs):
    return sum(x, **kargs)/tensor_size(x, **kargs)


def deriv(dep, indep):
    return dep.adjoints[indep]


def cross_entropy_multi(y, t, usebits=False, out_axes=()):
    logscale = np.float(1. / np.log(2.0) if usebits else 1.)
    return -sum(safelog(y) * t, out_axes=out_axes)*logscale


def cross_entropy_binary(y, t, out_axes=()):
    a = - safelog(y) * t
    b = - safelog(1 - y) * (1 - t)
    return sum(a + b, out_axes=out_axes)
    

class Function(NameableValue):
    
    def __init__(self, ops):
        super(Function, self).__init__()
        from geon.backends.graph.analysis import Digraph
        self.ops = Digraph(ops)
        use, defs = set(), set()
        for op in self.ops.topsort():
            #Kernel defines the def of each operation
            defs |= set([op])
            #Kernel uses the use of each operation
            #except whatever can be held in registers
            use |= set(op.inputs) - defs
        self.use = use
        self.defs = defs

    @property
    def graph_label(self):
        return self.__class__.__name__
        
    @property
    def inputs(self):
        return self.use
