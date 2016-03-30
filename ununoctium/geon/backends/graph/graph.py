from contextlib import contextmanager
import weakref

import geon.backends.graph.typing as typing
from geon.backends.graph.errors import *

import numpy as np


#TODO Probably don't need this separate from Op, particularly if variable goes away
class Arg(object):
    """
    An Arg is something that can appear as a Python function/operator argument, but might not be inserted directly
    into the graph.
    """

    @property
    def graph(self):
        raise NotImplementedError()

    def __iter__(self):
        return OpIterator(self.op)

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

    def __getitem__(self, val):
        print "Arg: %s" % (val,)
        return self

    @property
    def T(self):
        return transpose(self)


def posneg(x):
    s = .5*sig(x)

    return .5+s, .5-s

class GraphOp(Arg):
    def __init__(self, graph_type, out=None):
        self.graph_type = graph_type
        if out is None:
            self.__out = lambda : None
        else:
            self.out = out
        graph = Graph.get_default_graph()
        self._graph_ref = weakref.ref(graph)
        graph.add_op(self)
        # Name assigned by user
        self.name = None

        # Ops that directly use the result
        self.users = weakref.WeakSet()
        self.args = ()

    @property
    def out(self):
        return self.__out()

    @out.setter
    def out(self, value):
        if not value.graph_type.is_subtype_of(self.graph_type):
            raise IncompatibleTypesError()
        self.__out = weakref.ref(value)
        if value is not self:
            value.users.add(self)

    @property
    def ops(self):
        return []

    @staticmethod
    def as_op(x):
        if isinstance(x, GraphOp):
            return x

        return Constant(x)

    def arg_types(self):
        return (arg.graph_type for arg in self.args)

    @property
    def size(self):
        result = 1
        for d in self.graph_type.shape:
            result = result * d
        return result

    def __str__(self):
        return self.__class__.__name__

    @property
    def graph(self):
        return self._graph_ref()

    def generate_add_delta(self, adjoints, delta):
        if self not in adjoints:
            adjoints[self] = delta
        else:
            adjoints[self] = delta + adjoints[self]

    def evaluate(self, environment, *args):
        raise NotImplementedError()


class ControlOp(GraphOp):
    def __init__(self):
        super(ControlOp, self).__init__(typing.Void)


class TensorOp(GraphOp):
    """
    An Op is the result of some sort of operation.
    """
    def __init__(self, out, *args):
        opargs = tuple(GraphOp.as_op(arg) for arg in args)
        for arg in opargs:
            arg.users.add(self)
        graph_type = self.compute_graph_type(*(arg.graph_type for arg in opargs))
        if out is None:
            out = empty(*graph_type.array_args())

        super(TensorOp, self).__init__(graph_type=graph_type, out=out)
        self.args = opargs

    def compute_graph_type(self, *argtypes):
        raise NotImplementedError()

    def reshape(self, shape):
        return reshape(self, shape)


class ElementWise(TensorOp):
    def compute_graph_type(self, *argtypes):
        return typing.elementwise_graph_type(np.float32, *argtypes)


def c_strides(dtype, shape):
    stride = dtype.itemsize
    strides = []
    for l in shape:
        strides.append(stride)
        stride = stride*l
    return tuple(reversed(strides))


class ConstantScalarOp(GraphOp):
    def __init__(self, graph_type, *args):
        super(ConstantScalarOp, self).__init__(self, graph_type)


class AllocationTensorOp(GraphOp):
    def __init__(self, shape, dtype=None):
        super(AllocationTensorOp, self).__init__(typing.Array[shape, dtype])
        self.strides = c_strides(self.graph_type.dtype, self.graph_type.shape)
        self.out = self
        self.aliases = weakref.WeakSet()


class AliasOp(AllocationTensorOp):
    """
    Allocates a descriptor that aliases another allocation.
    """
    def __init__(self, shape, aliased):
        super(AliasOp, self).__init__(shape, aliased.graph_type.dtype)
        self.args = (aliased,)
        aliased.out.aliases.add(self)


class OpIterValue(TensorOp):
    def __init__(self, sequence):
        super(OpIterValue, self).__init__(sequence)


class OpIterator(GraphOp):
    def __init__(self, sequence):
        super(OpIterator, self).__init__(None, sequence)
        self.__next = True

    def __iter__(self):
        return OpIterator(*self.args)

    def next(self):
        if self.__next:
            self.__next = False
            sequence, = self.args
            return OpIterValue(sequence)
        else:
            raise StopIteration()


class input(AllocationTensorOp):
    """
    Can be set externally.
    """

    def __init__(self, shape, dtype=np.float32):
        super(input, self).__init__(shape, dtype)

    def evaluate(self, environment):
        return environment.input(self.name, self.graph_type)

    def generate_adjoints(self, tape, delta):
        pass


# Not sure if we'll need this
class variable(Arg):

    def __init__(self, name=None):
        self.__graph = Graph.get_default_graph()
        self.name = name
        self.__op = None
        self.__graph.variables[name] = self

    @property
    def graph(self):
        return self.__graph

    @property
    def op(self):
        if self.__op is None:
            raise UnititializedVariableError()
        return self.__op

    def set(self, op):
        self.__op = GraphOp.as_op(op)
        return self


class Constant(AllocationTensorOp):
    """
    A constant that appears in a graph.
    """
    def __init__(self, const):
        if isinstance(const, np.ndarray):
            shape =  const.shape
        else:
            shape = ()
        super(Constant, self).__init__(shape)
        self.const = const

    def evaluate(self, environment):
        return environment.constant(self.const)

    def generate_adjoints(self, tape, delta):
        pass

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.const)


class absolute(ElementWise):
    def __init__(self, x, out=None):
        super(absolute, self).__init__(out, x)

    def evaluate(self, environment, out, x):
        return environment.absolute(x, out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, sig(x)*delta)


class add(ElementWise):
    def __init__(self, x, y, out=None):
        super(add, self).__init__(out, x, y)

    def evaluate(self, environment, out, x, y):
        return environment.add(x, y, out)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, delta)


class cos(ElementWise):
    def __init__(self, x, out=None):
        super(cos, self).__init__(out, x)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*sin(x))

    def evaluate(self, environment, out, x):
        return environment.cos(x, out)


# This makes the derivative simpler if we need it
def divide(x, y, out=None):
    result = multiply(x, reciprocal(y), out=out)
    return result


#TODO Replace with a simple dot operation wrapped by the messy dimension cases
class dot(TensorOp):
    def __init__(self, x, y, out=None):
        xr = x
        if len(x.graph_type.shape) == 0:
            xr = reshape(x, (1,1))
        elif len(x.graph_type.shape) != 2:
            d1 = 1
            for d in x.graph_type.shape[:-1]:
                d1 = d1*d
            xr = reshape(x, (d1, x.graph_type.shape[-1]))
        yr = y
        if len(y.graph_type.shape) == 0:
            yr = reshape(y, (1,1))
        elif len(y.graph_type.shape) != 2:
            d1 = 1
            for d in y.graph_type.shape[1:]:
                d1 = d1*d
            yr = reshape(y, (y.graph_type.shape[0], d1))

        xshape = x.graph_type.shape
        yshape = y.graph_type.shape

        if len(xshape) == 0:
            self.shape = yshape
        elif len(yshape) == 0:
            self.shape = xshape
        elif xshape[-1] != yshape[0]:
            raise IncompatibleShapesError()
        else:
            self.shape = tuple(xshape[:-1]+yshape[1:])

        super(dot, self).__init__(out, xr, yr)

    def compute_graph_type(self, *argtypes):
        return typing.Array[self.shape, np.result_type(*(argtype.dtype for argtype in argtypes))]

    def evaluate(self, environment, out, x, y):
        return environment.dot(x, y, out)

    def generate_adjoints(self, adjoints, delta, x, y):

        dr = delta
        if len(dr.graph_type.shape) != 2:
            dr = reshape(delta, (x.graph_type.shape[0], y.graph_type.shape[1]))
        x.generate_add_delta(adjoints, dot(dr, y.T))
        y.generate_add_delta(adjoints, dot(x.T, dr))


class empty(AllocationTensorOp):
    def __init__(self, shape, dtype=None, name=None, persist_values=None, *args):
        super(empty, self).__init__(shape, dtype=dtype)
        self.name = name
        self.persist_values = persist_values
        self.other_args = args

    def generate_adjoints(self, adjoints, delta):
        pass

    def evaluate(self, environment):
        return environment.empty(*self.graph_type.array_args())


class exp(ElementWise):
    def __init__(self, x, out=None):
        super(exp, self).__init__(out, x)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta)

    def evaluate(self, environment, out, x):
        return environment.exp(x, out)


class log(ElementWise):
    def __init__(self, x, out=None):
        super(log, self).__init__(out, x)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta/x)

    def evaluate(self, environment, out, x):
        return environment.log(x, out)


class maximum(ElementWise):
    def __init__(self, x, y, out=None):
        super(maximum, self).__init__(out, x, y)

    def evaluate(self, environment, out, x, y):
        return environment.maximum(x, y, out=out)

    def generate_adjoins(self, adjoints, delta, x, y):
        p, n = posneg(x-y)
        x.generate_add_delta(delta*p)
        y.generate_add_delta(delta*n)


class minimum(ElementWise):
    def __init__(self, x, y, out=None):
        super(minimum, self).__init__(out, x, y)

    def evaluate(self, environment, out, x, y):
        return environment.minimum(x, y, out=out)

    def generate_adjoins(self, adjoints, delta, x, y):
        p, n = posneg(y-x)
        x.generate_add_delta(delta*p)
        y.generate_add_delta(delta*n)


class multiply(ElementWise):
    def __init__(self, x, y, out=None):
        super(multiply, self).__init__(out, x, y)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta*y)
        y.generate_add_delta(adjoints, x*delta)


    def evaluate(self, environment, out, x, y):
        return environment.multiply(x, y, out)


class negative(ElementWise):
    def __init__(self, x, out=None):
        super(negative, self).__init__(out, x)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -delta)

    def evaluate(self, environment, out, x):
        return environment.negative(x, out)


class ones(AllocationTensorOp):
    def __init__(self, shape, dtype=None, name=None, persist_values=None, *args):
        super(ones, self).__init__(shape, dtype=dtype)
        self.name = name
        self.persist_values = persist_values
        self.other_args = args

    def generate_adjoints(self, adjoints, delta):
        pass

    def evaluate(self, environment):
        return environment.ones(*self.graph_type.array_args())


class reciprocal(ElementWise):
    def __init__(self, x, out=None):
        super(reciprocal, self).__init__(out, x)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -self*self*delta)

    def evaluate(self, environment, out, x):
        return environment.reciprocal(x, out)


#TODO This should be restride, as should transpose, is terms of (i,j,k) -> ((i,j),k) i.e. remap
class reshape(AliasOp):
    def __init__(self, x, shape):
        super(reshape, self).__init__(shape, x)
        if self.size != x.size:
            raise ValueError('total size of new array must be unchanged')

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, reshape(delta, x.graph_type.shape))

    def evaluate(self, environment, x):
        return environment.reshape(x, self.graph_type.shape)


class sig(ElementWise):
    def __init__(self, x, out=None):
        super(sig, self).__init__(out, x)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*self*(1.0-self))

    def evaluate(self, environment, out, x):
        return environment.sig(x, out)

class sin(ElementWise):
    def __init__(self, x, out=None):
        super(sin, self).__init__(out, x)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*cos(x))

    def evaluate(self, environment, out, x):
        return environment.sin(x, out)


class sqrt(ElementWise):
    def __init__(self, x, out=None):
        super(sqrt, self).__init__(out, x)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, .5*delta*self)

    def evaluate(self, environment, out, x):
        return environment.sqrt(x, out)


class square(ElementWise):
    def __init__(self, x, out=None):
        super(square, self).__init__(out, x)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, 2.0*delta*x)

    def evaluate(self, environment, out, x):
        return environment.square(x, out)


class subtract(ElementWise):
    def __init__(self, x, y, out=None):
        super(subtract, self).__init__(out, x, y)

    def evaluate(self, value, x, y):
        return x-y

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, -delta)

    def evaluate(self, environment, out, x, y):
        return environment.subtract(x, y, out)


class tanh(ElementWise):
    def __init__(self, x, out=None):
        super(tanh, self).__init__(out, x)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*(1.0-self*self))

    def evaluate(self, environment, out, x):
        return environment.tanh(x, out)


class transpose(AliasOp):
    def __init__(self, x):
        super(transpose, self).__init__(tuple(reversed(x.graph_type.shape)), x)

    def evaluate(self, environment, x):
        return environment.transpose()

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta.T)


class zeros(AllocationTensorOp):
    def __init__(self, shape, dtype=None, name=None, persist_values=None, *args):
        super(zeros, self).__init__(shape, dtype=dtype)
        self.name = name
        self.persist_values = persist_values
        self.other_args = args

    def generate_adjoints(self, adjoints, delta):
        pass

    def evaluate(self, environment):
        return environment.zeros(*self.graph_type.array_args())


class range(GraphOp):
    def __init__(self, start, stop=None, step=1):
        if stop is None:
            start = 0
            stop = start
        super(range, self).__init__(None, start, stop, step)


def deriv(dep, indep):
    return Graph.get_default_graph().get_adjoints(dep)[indep]


class ControlBlock(object):

    def __init__(self):
        self.__ops = []


    def add_context_op(self, op):
        self.__ops.append(op)

    @property
    def ops(self):
        return self.__ops


class NestedControlBlock(ControlBlock, ControlOp):
    def __init__(self, context):
        super(NestedControlBlock, self).__init__()
        self.parent_context = context


class Iterator(NestedControlBlock):
    def __init__(self, context):
        super(Iterator, self).__init__(context)


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
            if op.out is not op:
                outid = '=>%d' % (ids[op.out],)
            print '%d:%s%s:%s%s%s' % (ids[op], name, op.graph_type.shape, op, tuple(ids[arg] for arg in op.args), outid)
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


class VariableBlock(object):
    def __setattr__(self, name, value):
        value.name = name
        super(VariableBlock, self).__setattr__(name, value)


class Graph(object):

    def __init__(self):
        super(Graph, self).__init__()
        self.root_context = ControlBlock()
        self.context = self.root_context
        self.inputs = {}
        self.variables = VariableBlock()
        self.op_adjoints = {}

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
        self.context.add_context_op(op)

    @staticmethod
    def get_ordered_ops(op, ordered_ops):
        if op not in ordered_ops:
            for arg in op.args:
                Graph.get_ordered_ops(arg, ordered_ops)
            ordered_ops.append(op)

    def get_adjoints(self, op):
        if op in self.op_adjoints:
            return self.op_adjoints[op]
        adjoints = {}
        ordered_ops = []
        Graph.get_ordered_ops(op, ordered_ops)
        self.op_adjoints[op] = adjoints
        adjoints[op] = ones(*op.graph_type.array_args())
        for o in reversed(ordered_ops):
            o.generate_adjoints(adjoints, adjoints[o], *o.args)
        return adjoints

    def ordered_ops(self):
        ops = []

        def addops(g):
            ops.append(g)
            for op in g.ops:
                addops(op)

        addops(self.root_context)

        return ops

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

    def sgn(self, a, out=None):
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



