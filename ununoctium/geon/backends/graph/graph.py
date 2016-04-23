from contextlib import contextmanager
import weakref

from geon.backends.graph.names import AxisGenerator, NameableValue, VariableBlock, merge_axes
import geon.backends.graph.typing as typing
from geon.backends.graph.errors import *

import numpy as np


class GraphMetaclass(type):
    """Ensures that there is a default graph while running __init__"""
    def __new__(cls, name, parents, attrs):
        return super(GraphMetaclass, cls).__new__(cls, name, parents, attrs)

    def __call__(cls, *args, **kargs):
        gr = Graph()
        with default_graph(gr) as g:
            return super(GraphMetaclass, cls).__call__(*args, graph=gr, **kargs)


class GraphComponent(object):
    """
    Superclass for all models.

    Ensures that __metaclass__ is set.
    """
    __metaclass__ = GraphMetaclass

    def __init__(self, graph, **kargs):
        super(GraphComponent, self).__init__(**kargs)
        self.graph = graph
        self.a = AxisGenerator('a')
        self.params = graph.variables


class Model(GraphComponent):
    def __init__(self, **kargs):
        super(Model, self).__init__(**kargs)


def posneg(x):
    s = .5*sgn(x)

    return .5+s, .5-s


class GraphOp(NameableValue):
    """Any operation that can be in a computation graph"""

    def __pre_init_before_super__(self, **kwds):
        """
        These inits are run before __init__ in base to super order.

        Each method should end by returning a call to its super, i.e.
        def __pre_init_before_super__(self, arg1, ..., **kwds):
            do stuff
            return super(MyClass, self).__pre_init_before_super__(**kwds)
        """
        return kwds

    def __pre_init_after_super__(self, **kwds):
        """
        These inits are run before __init__ in super to base order.

        Each method should begin with a call to its super, and return that value
        def __pre_init_after_super__(self, arg1, ..., **kwds):
            kwds = super(MyClass, self).__pre_init_before_super__(**kwds)
            do stuff
            return kwds
        """
        for arg in self.inputs:
            arg.users.add(self)

        return kwds

    def __init__(self, **kwds):
        graph = Graph.get_default_graph()
        self._graph_ref = weakref.ref(graph)
        self._context_ref = weakref.ref(graph.context)
        self.predecessors = weakref.WeakSet()

        kwds = self.__pre_init_before_super__(**kwds)
        kwds = self.__pre_init_after_super__(**kwds)
        super(GraphOp, self).__init__(**kwds)
        self.opid = self.graph.add_op(self)

    @property
    def graph(self):
        return self._graph_ref()

    @property
    def context(self):
        return self._context_ref()

    @property
    def inputs(self):
       return ()

    @staticmethod
    def as_op(x):
        if isinstance(x, ValueOp):
            return x

        return Constant(x)

    @property
    def ops(self):
        return []

    def evaluate(self, environment, *args):
        raise NotImplementedError()

    def __str__(self):
        return '<{cl} {opid}>'.format(cl=self.__class__.__name__, opid=self.opid)


class ControlOp(GraphOp):
    def __init__(self, **kargs):
        super(ControlOp, self).__init__(**kargs)


class RandomStateOp(GraphOp):
    def __init__(self, seed=None, **kargs):
        super(RandomStateOp, self).__init__(**kargs)
        self.seed = seed


class ValueOp(GraphOp):
    def __pre_init_before_super__(self, graph_type=None, **kwds):
        if graph_type is None:
            graph_type = self.compute_graph_type(*(arg.graph_type for arg in self.inputs))

        self.graph_type = graph_type
        return super(ValueOp, self).__pre_init_before_super__(**kwds)

    def __init__(self, **kwds):
        super(ValueOp, self).__init__(**kwds)

        # Ops that directly use the result
        self.users = weakref.WeakSet()  # Name assigned by user

    @property
    def output(self):
        return self

    @property
    def size(self):
        result = 1
        for d in self.graph_type.shape:
            result = result * d
        return result

    def generate_add_delta(self, adjoints, delta):
        if self not in adjoints:
            adjoints[self] = delta
        else:
            adjoints[self] = delta + adjoints[self]

    def reshape(self, shape):
        return reshape(self, shape)

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

    @property
    def T(self):
        return transpose(self)


class ArgsOp(GraphOp):
    def __pre_init_before_super__(self, args=(), **kargs):
        self.__args = tuple(GraphOp.as_op(arg) for arg in args)
        return super(ArgsOp, self).__pre_init_before_super__(**kargs)

    def __init__(self, **kargs):
        super(ArgsOp, self).__init__(**kargs)

    @property
    def inputs(self):
        return self.__args

    def add_dependencies(self):
        super(ArgsOp, self).add_dependencies()
        for arg in self.inputs:
            arg.users.add(self)


class ComputationOp(ArgsOp, ValueOp):
    """
    An TensorOp is the result of some sort of operation.
    """
    def __init__(self, **kargs):
        super(ComputationOp, self).__init__(**kargs)

    def compute_graph_type(self, *argtypes):
        raise NotImplementedError()

    def add_dependencies(self):
        self.output.users.add(self)

    @property
    def output(self):
        return self.__out()

    @output.setter
    def output(self, value):
        if not value.graph_type.is_subtype_of(self.graph_type):
            raise IncompatibleTypesError()
        self.__out = weakref.ref(value)
        if value is not self:
            value.users.add(self)

    def arg_types(self):
        return (arg.graph_type for arg in self.inputs)

class OutputArgOp(ComputationOp):
    """
    An OutputArgOp has an out= argument for its result.
    """

    def __pre_init_after_super__(self, out=None, **kargs):
        kargs = super(OutputArgOp, self).__pre_init_after_super__(**kargs)
        if out is None:
            self.output = empty(**self.graph_type.array_args())
        else:
            self.output = out
        return kargs

    def __init__(self, **kargs):
        super(OutputArgOp, self).__init__(**kargs)


class Component(GraphComponent, OutputArgOp):
    def __init__(self, **kargs):
        super(Component, self).__init__(**kargs)


class ElementWise(OutputArgOp):
    def __init__(self, **kargs):
        super(ElementWise, self).__init__(**kargs)

    def compute_graph_type(self, *argtypes):
        return typing.elementwise_graph_type(np.float32, *argtypes)


def c_strides(dtype, shape):
    stride = dtype.itemsize
    strides = []
    for l in shape:
        strides.append(stride)
        stride = stride*l
    return tuple(reversed(strides))


class AllocationOp(ValueOp):
    def __init__(self, axes=None, dtype=None, **kargs):
        super(AllocationOp, self).__init__(graph_type=typing.Array[axes, dtype], **kargs)
        self.axes = axes
        self.aliases = weakref.WeakSet()


class AliasOp(ArgsOp, AllocationOp):
    """
    Allocates a descriptor that aliases another allocation.
    """
    def __init__(self, axes, aliased, **kargs):
        super(AliasOp, self).__init__(axes=axes, dtype=aliased.graph_type.dtype, args=(aliased,), **kargs)
        aliased.output.aliases.add(self)

    @property
    def aliased(self):
        return self.inputs[0]


class input(AllocationOp):
    """
    Can be set externally.
    """
    def __init__(self, axes, **kargs):
        super(input, self).__init__(axes=axes, **kargs)

    def evaluate(self, environment):
        return environment.input(self.name, self.graph_type)

    def generate_adjoints(self, tape, delta):
        pass


class Constant(AllocationOp):
    """
    A constant that appears in a graph.
    """
    def __init__(self, const, **kargs):
        if isinstance(const, np.ndarray):
            super(Constant, self).__init__(shape=const.shape, dtype=const.dtype, **kargs)
        else:
            super(Constant, self).__init__((), **kargs)
        self.const = const

    def evaluate(self, environment):
        return environment.constant(self.const)

    def generate_adjoints(self, tape, delta):
        pass

    def __str__(self):
        return '<{cl} {opid} ({const})>'.format(cl=self.__class__.__name__, opid=self.opid, const=self.const)


class absolute(ElementWise):
    def __init__(self, x, out=None):
        super(absolute, self).__init__(out=out, args=(x,))

    def evaluate(self, environment, out, x):
        return environment.absolute(x, out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, sig(x)*delta)


class add(ElementWise):
    def __init__(self, x, y, out=None):
        super(add, self).__init__(out=out, args=(x, y))

    def evaluate(self, environment, out, x, y):
        return environment.add(x, y, out)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, delta)


class cos(ElementWise):
    def __init__(self, x, out=None):
        super(cos, self).__init__(out=out, args=(x,))

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*sin(x))

    def evaluate(self, environment, out, x):
        return environment.cos(x, out)


# This makes the derivative simpler if we need it
def divide(x, y, out=None):
    result = multiply(x, reciprocal(y), out=out)
    return result


class dot(OutputArgOp):
    def __init__(self, x, y, out=None):
        lcr = merge_axes(x.graph_type.axes, y.graph_type.axes)
        self.axes = lcr[0] + lcr[-1]
        super(dot, self).__init__(out=out, args=(x, y))

    def compute_graph_type(self, *argtypes):
        return typing.Array[self.axes, np.result_type(*(argtype.dtype for argtype in argtypes))]

    def evaluate(self, environment, out, x, y):
        return environment.dot(x, y, out)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, dot(delta, y))
        y.generate_add_delta(adjoints, dot(x, delta))


class empty(AllocationOp):
    def __init__(self, **kargs):
        super(empty, self).__init__(**kargs)

    def generate_adjoints(self, adjoints, delta):
        pass

    def evaluate(self, environment):
        return environment.empty(**self.graph_type.array_args())


class exp(ElementWise):
    def __init__(self, x, out=None):
        super(exp, self).__init__(out=out, args=(x,))

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta)

    def evaluate(self, environment, out, x):
        return environment.exp(x, out)


class log(ElementWise):
    def __init__(self, x, out=None):
        super(log, self).__init__(out=out, args=(x,))

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta/x)

    def evaluate(self, environment, out, x):
        return environment.log(x, out)


class maximum(ElementWise):
    def __init__(self, x, y, out=None):
        super(maximum, self).__init__(out=out, args=(x, y))

    def evaluate(self, environment, out, x, y):
        return environment.maximum(x, y, out=out)

    def generate_adjoints(self, adjoints, delta, x, y):
        p, n = posneg(x-y)
        x.generate_add_delta(delta*p)
        y.generate_add_delta(delta*n)


class minimum(ElementWise):
    def __init__(self, x, y, out=None):
        super(minimum, self).__init__(out=out, args=(x, y))

    def evaluate(self, environment, out, x, y):
        return environment.minimum(x, y, out=out)

    def generate_adjoints(self, adjoints, delta, x, y):
        p, n = posneg(y-x)
        x.generate_add_delta(delta*p)
        y.generate_add_delta(delta*n)


class multiply(ElementWise):
    def __init__(self, x, y, out=None):
        super(multiply, self).__init__(out=out, args=(x, y))

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta*y)
        y.generate_add_delta(adjoints, x*delta)


    def evaluate(self, environment, out, x, y):
        return environment.multiply(x, y, out)


class negative(ElementWise):
    def __init__(self, x, out=None):
        super(negative, self).__init__(out=out, args=(x,))

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -delta)

    def evaluate(self, environment, out, x):
        return environment.negative(x, out)


class ones(AllocationOp):
    def __init__(self, **kargs):
        super(ones, self).__init__(**kargs)

    def generate_adjoints(self, adjoints, delta):
        pass

    def evaluate(self, environment):
        return environment.ones(**self.graph_type.array_args())


class reciprocal(ElementWise):
    def __init__(self, x, out=None):
        super(reciprocal, self).__init__(out=out, args=(x,))

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -self*self*delta)

    def evaluate(self, environment, out, x):
        return environment.reciprocal(x, out)


#TODO This should be restride, as should transpose, is terms of (i,j,k) -> ((i,j),k) i.e. remap
class reshape(AliasOp):
    def __init__(self, x, shape):
        super(reshape, self).__init__(shape=shape, aliased=x)
        if self.size != x.size:
            raise ValueError('total size of new array must be unchanged')

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, reshape(delta, x.graph_type.shape))

    def evaluate(self, environment, x):
        return environment.reshape(x, self.graph_type.shape)


class sgn(ElementWise):
    def __init__(self, x, out=None):
        super(sgn, self).__init__(out=out, args=(x,))

    def generate_adjoints(self, adjoints, delta, x):
        # Zero
        pass

    def evaluate(self, environment, out, x):
        return environment.sign(x, out)


class sig(ElementWise):
    def __init__(self, x, out=None):
        super(sig, self).__init__(out=out, args=(x,))

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*self*(1.0-self))

    def evaluate(self, environment, out, x):
        return environment.sig(x, out)

class sin(ElementWise):
    def __init__(self, x, out=None):
        super(sin, self).__init__(out=out, args=(x,))

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*cos(x))

    def evaluate(self, environment, out, x):
        return environment.sin(x, out)


class sqrt(ElementWise):
    def __init__(self, x, out=None):
        super(sqrt, self).__init__(out=out, args=(x,))

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, .5*delta*self)

    def evaluate(self, environment, out, x):
        return environment.sqrt(x, out)


class square(ElementWise):
    def __init__(self, x, out=None):
        super(square, self).__init__(out=out, args=(x,))

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, 2.0*delta*x)

    def evaluate(self, environment, out, x):
        return environment.square(x, out)


class subtract(ElementWise):
    def __init__(self, x, y, out=None):
        super(subtract, self).__init__(out=out, args=(x, y))

    def evaluate(self, value, x, y):
        return x-y

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, -delta)

    def evaluate(self, environment, out, x, y):
        return environment.subtract(x, y, out)


class tanh(ElementWise):
    def __init__(self, x, out=None):
        super(tanh, self).__init__(out=out, args=(x,))

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*(1.0-self*self))

    def evaluate(self, environment, out, x):
        return environment.tanh(x, out)


class transpose(AliasOp):
    def __init__(self, x):
        super(transpose, self).__init__(axes=tuple(reversed(x.graph_type.axes)), aliased=x)

    def evaluate(self, environment, x):
        return environment.transpose(x)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta.T)


class zeros(AllocationOp):
    def __init__(self, **kargs):
        super(zeros, self).__init__(**kargs)

    def generate_adjoints(self, adjoints, delta):
        pass

    def evaluate(self, environment):
        return environment.zeros(**self.graph_type.array_args())


class range(ValueOp):
    def __init__(self, start, stop=None, step=1):
        if stop is None:
            start = 0
            stop = start
        super(range, self).__init__(args=(start, stop, step))


def deriv(dep, indep):
    return Graph.get_default_graph().get_adjoints(dep)[indep]


class ControlBlock(object):

    def __init__(self, last_op=None, **kargs):
        super(ControlBlock, self).__init__(**kargs)
        # Ops in this block
        self.__ops = []
        # Predecessor of this block
        self._last_op = last_op

    def add_context_op(self, op):
        if self._last_op is not None:
            op.predecessors.add(self._last_op)
        self._last_op = op
        self.__ops.append(op)

    @property
    def ops(self):
        return self.__ops


class NestedControlBlock(ControlBlock, ControlOp):
    def __init__(self, context, **kargs):
        super(NestedControlBlock, self).__init__(**kargs)
        self.parent_context = context


class Iterator(NestedControlBlock):
    def __init__(self, **kargs):
        super(Iterator, self).__init__(**kargs)


def show_graph(g):
    ids = {}

    def opids(g):
        ids[g] = len(ids)
        for op in g.ops:
            opids(op)

    opids(g.root_context)

    def show_op(g):
        for op in g.ops:
            name = ''
            if op.name is not None:
                name = op.name
            outid = ''
            if op.output is not op:
                outid = '=>%d' % (ids[op.output],)

            print '%d:%s%s:%s%s%s' % (ids[op], name, op.graph_type.axes, op, tuple(ids[arg] for arg in op.inputs), outid)
            show_op(op)
    show_op(g.root_context)

@contextmanager
def default_graph(graph=None):
    thread_data = Graph.get_thread_data()
    old_graph = thread_data.graph
    if graph is None:
        graph=Graph()
    try:
        thread_data.graph = graph
        yield(graph.variables)
    finally:
        thread_data.graph = old_graph


@contextmanager
def iterate(iterable):
    graph = Graph.get_default_graph()
    old_context = graph.context
    items = iter(iterable)
    n = items.next()
    iterator = Iterator(old_context)
    with Graph.context(iterator):
        if isinstance(n, tuple):
            var = tuple(graph.variable() for v in n)
            for vv, v in zip(var, n):
                vv.set(v)
        else:
            var = graph.variable()
            var.set(n)
        yield (var)


class Graph(object):

    def __init__(self, **kargs):
        super(Graph, self).__init__(**kargs)
        self.root_context = ControlBlock()
        self.context = self.root_context
        self.inputs = {}
        self.variables = VariableBlock()
        self.op_adjoints = {}
        self.ordered_ops = []

    import threading
    __thread_data = threading.local()
    __thread_data.graph = None

    @staticmethod
    def get_thread_data():
        return Graph.__thread_data

    @staticmethod
    def get_default_graph():
        graph = Graph.__thread_data.graph

        if graph is None:
            # TODO: Work-around for working with Neon
            import neon
            be = neon.NervanaObject.be
            if be is not None and hasattr(be, 'gr'):
                return be.gr
            raise MissingGraphError()

        return graph

    @contextmanager
    def bound_context(self, context):
        old_context = self.context
        try:
            self.context = context
            yield()
        finally:
            self.context = old_context

    def add_op(self, op):
        opid = np.int64(len(self.ordered_ops))
        self.ordered_ops.append(op)
        self.context.add_context_op(op)
        return opid

    @staticmethod
    def get_ordered_ops(op, ordered_ops):
        """
        Get dependent ops ordered for autodiff.
        """
        if op not in ordered_ops:
            if isinstance(op, ArgsOp):
                for arg in op.inputs:
                    Graph.get_ordered_ops(arg, ordered_ops)
            ordered_ops.append(op)

    def get_adjoints(self, op):
        if op in self.op_adjoints:
            return self.op_adjoints[op]
        adjoints = {}
        ordered_ops = []
        Graph.get_ordered_ops(op, ordered_ops)
        self.op_adjoints[op] = adjoints
        adjoints[op] = ones(**op.graph_type.array_args())
        for o in reversed(ordered_ops):
            o.generate_adjoints(adjoints, adjoints[o], *o.inputs)
        return adjoints

    def analyze_liveness(self, results):
        liveness = [set() for op in self.ordered_ops]
        i = len(liveness)-1
        for result in results:
            liveness[i].add(result.output)
        while i > 0:
            op = self.ordered_ops[i]
            prealive = liveness[i-1]
            alive = set(liveness[i])
            if isinstance(op, ValueOp):
                output = op.output
                alive.discard(output)
                for arg in op.inputs:
                    alive.add(arg.output)
                prealive |= alive
            i = i-1
        self.liveness = liveness
        return liveness



    # Neon backend functions
    # TODO These are just here as placeholders
    def add_fc_bias(self, inputs, bias):
        pass

    def argmax(self, axis=None, out=None, keepdims=None):
        pass

    def array(self, ary,  dtype=None, name=None, persist_values=None, *args):
        pass

    def batched_dot(self, A, B, C, alpha=None, beta=None, relu=None):
        pass

    def begin(self, block, identifier):
        pass

    def bprop_conv(self, layer, F, E, grad_I, alpha=None, repeat=None):
        pass

    def bprop_pool(self, layer, I, E, grad_I):
        pass

    def check_cafe_compat(self):
        pass

    def clip(self, a, a_min, a_max, out=None):
        pass

    def compound_bprop_lut(self, nin, inputs, error, *args):
        pass

    def compound_dot(self, A, B, C, alpha=None, beta=None, relu=None):
        pass

    def conv_layer(self, dtype, N, C, K, D=None, H=None, W=None, T=None, R=None, *args):
        pass

    def deconv_layer(self, dtype, N, C, K, P, Q, R=None, S=None, *args):
        pass

    def empty_like(self, other_ary, name=None, persist_values=None):
        pass

    def end(self, block, identifier):
        pass

    def equal(self, a, b, out=None):
        pass

    def exp2(self, a, out=None):
        pass

    def fabs(self, a, out=None):
        pass

    def finite(self, a, out=None):
        pass

    def fprop_conv(self, layer, I, F, O, alpha=None, relu=None, repeat=None):
        pass

    def fprop_pool(self, layer, I, O):
        pass

    def gen_rng(self, seed=None):
        pass

    def greater(self, a, b, out=None):
        pass

    def greater_equal(self, a, b, out=None):
        pass

    def less(self, a, b, out=None):
        pass

    def less_equal(self, a, b, out=None):
        pass

    def log2(self, a, out=None):
        pass

    def make_binary_mask(self, out, keepthresh=None):
        pass

    def max(self, axis=None, out=None, keepdims=None):
        pass

    def mean(self, a, axis=None, partial=None, out=None, keepdims=None):
        pass

    def min(self, a, axis=None, out=None, keepdims=None):
        pass

    def not_equal(self, a, b, out=None):
        pass

    def onehot(self, indices, axis, out=None):
        pass

    def output_dim(self, X, S, padding, strides, pooling=None):
        pass

    def pool_layer(self, dtype, op, N, C, D=None, H=None, W=None, J=None, T=None, *args):
        pass

    def power(self, a, b, out=None):
        pass

    def revert_tensor(self, tensor):
        pass

    def rng_get_state(self, state):
        pass

    def rng_reset(self):
        pass

    def rng_set_state(self, state):
        pass

    def safelog(a, out=None):
        pass

    def set_caffe_compat(self):
        pass

    def sig2(self, a, out=None):
        pass

    def std(self, a, axis=None, partial=None, out=None, keepdims=None):
        pass

    def sum(self, a, axis=None, out=None, keepdims=None):
        pass

    def take(self, a, indices, axis, out=None):
        pass

    def tanh2(self, a, out=None):
        pass

    def true_divide(self, a, b, out=None):
        pass

    def update_conv(self, layer, I, E, grad_F, alpha=None, repeat=None):
        pass

    def update_fc_bias(self, err, out):
        pass

    def var(self, a, axis=None, partial=None, out=None, keepdims=None):
        pass

    def zeros_like(self, other_ary, name=None, persist_values=None):
        pass



