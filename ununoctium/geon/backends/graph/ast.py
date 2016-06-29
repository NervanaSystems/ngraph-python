from contextlib import contextmanager
import weakref
import numbers

import numpy as np

import neon.initializers.initializer as initializer

from geon.backends.graph.names import NameableValue
from geon.backends.graph.nodes import Node
import geon.backends.graph.typing as typing
from geon.backends.graph.errors import *
from geon.backends.graph.environment import get_current_environment, get_current_ops
import geon.backends.graph.arrayaxes as arrayaxes
from geon.backends.graph.arrayaxes import tensor_axes, get_batch_axes, set_batch_axes

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

    def evaluate(self, evaluator, out, *args):
        """Process op"""
        pass

    def __str__(self):
        return '<{cl}:{id}>'.format(cl=self.__class__.__name__, id=id(self))


class TensorAxesInfo(object):
    """Information about a use of a tensor with axes"""
    def __init__(self, axes, alloc=None, init=None, read_only=False, dtype=None, **kargs):
        super(TensorAxesInfo, self).__init__(**kargs)
        axes = tuple(axes)
        self.axes = axes
        self.views = weakref.WeakValueDictionary()
        self.alloc = alloc
        self.init = init
        self.read_only = read_only
        self.dtype = dtype

    def allocation_info(self, environment, axes=None, dtype=None, **kargs):
        if axes is None:
            axes = self.axes
        if dtype is None:
            dtype = self.dtype
        allocation_info = TensorAllocationInfo(axes=axes, dtype=dtype, **kargs)
        environment.set_tensor_allocation_info(self, allocation_info)
        return allocation_info

    def generate_initializations(self, tensor):
        if self.init:
            self.init.fill(tensor)

    def allocate(self, evaluator):
        if self.alloc is not None:
            allocation_info = self.allocation_info()
            self.alloc(evaluator)

    def get_or_default(self, axes, default_function):
        axes = tuple(axes)
        if self.views.has_key(axes):
            return self.views[axes]
        result = default_function()
        self.views[axes] = result
        return result

    def reaxe(self, reaxe):
        return self.get_or_default(reaxe, lambda : TensorReaxeViewInfo(tensor_axes_info=self, reaxe=reaxe, idx=len(self.views)))


class TensorViewInfo(object):
    """The use of a view of a tensor with axes"""
    def __init__(self, tensor_axes_info, idx, **kargs):
        super(TensorViewInfo, self).__init__(**kargs)
        self.tensor_axes_info = tensor_axes_info
        self.idx = idx

    def get_allocation(self, evaluator, tensor_allocation_info):
        try:
            return evaluator.environment.get_tensor_allocation_value(self)
        except KeyError:
            tensor = self.allocate(evaluator, tensor_allocation_info)
            evaluator.environment.set_tensor_allocation_value(self, tensor)
            return tensor

    def allocate(self, evaluator, tensor_allocation_info):
        raise NotImplementedError()


class TensorReaxeViewInfo(TensorViewInfo):
    """The use of a reaxe view of a tensor with axes"""
    def __init__(self, reaxe, **kargs):
        super(TensorReaxeViewInfo, self).__init__(**kargs)
        self.reaxe = reaxe

    def allocate(self, evaluator, tensor_allocation_info):
        return evaluator.reaxe(tensor_allocation_info, self.reaxe)


class TensorAllocationInfo(object):
    """Axes information about an allocated tensor"""
    def __init__(self, axes, dtype, sizes=None, strides=None, **kargs):
        super(TensorAllocationInfo, self).__init__(**kargs)
        self.axes = arrayaxes.canonicalize_axes(axes)
        self.dtype = dtype
        if sizes is None:
            sizes = tuple(arrayaxes.axes_sizes(self.axes))
        self.sizes = sizes
        if strides is None:
            strides = arrayaxes.c_axis_strides(dtype, axes)
        self.strides = strides
        self.axis_strides = dict()
        for axis, stride in zip(self.axes, self.strides):
            self.axis_strides[axis] = stride

    def reaxe_params(self, reaxes, broadcast=False):
        return arrayaxes.ReaxeParams(reaxes, self.axis_strides, broadcast)


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

    def __ne__(self, val):
        return not_equal(self, val)

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
        return get_current_environment().get_tensor_value(self)

    @property
    def tensor_axes_info(self):
        if self.__tensor_axes_info is None:
            self.__tensor_axes_info = self.compute_tensor_axes_info()
        return self.__tensor_axes_info

    def compute_tensor_axes_info(self):
        dtype = None
        if self.graph_type is not None:
            dtype = self.graph_type.dtype
        return TensorAxesInfo(self.axes.value, dtype=dtype)

    @property
    def call_info(self):
        if self.__call_info is None:
            self.__call_info = self.compute_call_info()
        return self.__call_info

    def compute_call_info(self):
        return [self.reaxe(self.resolved_axes)]

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


class decrement(VoidOp):
    def __init__(self, parameter, change, **kargs):
        super(decrement, self).__init__(out=parameter, args=(parameter, change), **kargs)

    def evaluate(self, evaluator, out, parameter, change):
        evaluator.update(parameter, change)


class SetItem(VoidOp):
    def __init__(self, tensor, item, val, **kargs):
        super(SetItem, self).__init__(args=(tensor, val), out=tensor, **kargs)
        self.item = item

    def evaluate(self, evaluator, out, tensor, val):
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
    def __init__(self, axes=None, init=None, dtype=np.float32, **kargs):
        super(AllocationOp, self).__init__(graph_type=typing.Array[AxesComp.as_axes(axes), dtype], **kargs)
        self.aliases = weakref.WeakSet()
        self.tensor_axes_info.init = init
        self.init = init

    @property
    def axes(self):
        return self.graph_type.axes


class placeholder(AllocationOp):
    """
    Can be set externally.
    """
    def __init__(self, **kargs):
        super(placeholder, self).__init__(**kargs)
        self.__axes = AxesAppendComp(AxesComp.as_axes(arrayaxes.get_phase_axes()), ValueAxesComp(self))

    def __axes__(self):
        return self.__axes

    def generate_adjoints(self, tape, delta):
        pass

    @property
    def value(self):
        return super(placeholder, self).value

    @value.setter
    def value(self, value):
        environment = get_current_environment()
        environment.set_placeholder_value(self, value)


class ConstantInit(VoidOp):
    def __init__(self, tensor, const, **kargs):
        super(ConstantInit, self).__init__(args=(tensor,), **kargs)
        self.const = const

    def evaluate(self, evaluator, out, tensor):
        self.evaluator.fill(out, self.const)


class Constant(AllocationOp):
    """
    A constant that appears in a graph.
    """
    def __init__(self, const, **kargs):
        super(Constant, self).__init__(axes=(), dtype=np.dtype(type(const)), init=self, **kargs)
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


    def evaluate(self, evaluator, out, x):
        max_axes = self.max_axes.value
        evaluator.argmax(x, max_axes, out)

    @property
    def axes(self):
        return AxesSubComp(tensor_axes(self.inputs[0]), self.max_axes)


class argmin(ComputationOp):
    def __init__(self, x, min_axes=None, **kargs):
        if min_axes is None:
            min_axes = tensor_sample_axes
        self.min_axes = AxesComp.as_axes(min_axes)
        super(argmin, self).__init__(args=(x,), dtype=np.int64, **kargs)


    def evaluate(self, evaluator, out, x):
        min_axes = self.min_axes.value
        evaluator.argmin(x, min_axes, out)

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

        super(dot, self).__init__(args=(x, y), **kargs)

    def evaluate(self, evaluator, out, x, y):
        resolved_reduction_axes = self.reduction_axes.value
        evaluator.dot(x, y, resolved_reduction_axes, out)

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
        super(softmax, self).__init__(args=(x,), **kargs)

    def evaluate(self, evaluator, out, x):
        evaluator.softmax(x, self.batch_axes.value, out)

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
        resolved_reduction_axes = self.reduction_axes.value
        evaluator.sum(x, resolved_reduction_axes, out)

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

    def evaluate(self, evaluator, out, x):
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

    @property
    def value(self):
        return get_current_environment()[self]


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


def cross_entropy_multi(y, t, usebits=False):
    logscale = np.float(1. / np.log(2.0) if usebits else 1.)
    return -sum(safelog(y) * t)*logscale


def cross_entropy_binary(y, t):
    a = - safelog(y) * t
    b = - safelog(1 - y) * (1 - t)
    return sum(a + b)
