from contextlib import contextmanager
import weakref

import numpy as np

class Error(Exception):
    """
    Base class for graph errors.
    """
    pass


class MissingGraphError(Error):
    """
    Graph cannot be determined.
    """

class UnititializedVariableError(Error):
    """
    Attempt to use the value of an unitialized variable.
    """

class IncompatibleShapesError(Error):
    """
    Incompatible shapes.
    """

#TODO Probably don't need this separate from Op, particularly if variable goes away
class Arg(object):
    """
    An Arg is something that can appear as a Python function/operator argument, but might not be inserted directly
    into the graph.
    """

    @property
    def op(self):
        return self.__graph()

    @property
    def graph(self):
        raise NotImplementedError()

    def __iter__(self):
        return OpIterator(self.graph, self.op)

    # Magic methods for builtin operations we want to use for creating nodes
    def __neg__(self):
        return negative(self.graph, self)

    def __pos__(self):
        return self

    def __abs__(self):
        return Abs(self)

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
        return Pow(self, val)

    @property
    def T(self):
        return transpose(self)


class Op(Arg):
    """
    An Op is the result of some sort of operation.
    """
    def __init__(self, *args):
        self.context = Graph.get_default_graph(args).context
        self.args = tuple(Op.as_op(arg) for arg in args)
        self.context.add_op(self)
        self.shape = None
        self.name = None
        self.users = weakref.WeakSet()
        self.out = None
        for arg in self.args:
            arg.users.add(self)

    def setup_out(self, out):
        if out is None:
            self.out = empty(self.shape)
        else:
            if out.shape != self.shape:
                raise IncompatibleShapesError()
            self.out = out

    @staticmethod
    def as_op(x):
        if isinstance(x, Arg):
            return x.op

        return Constant(x)

    @property
    def graph(self):
        return self.context.graph

    @property
    def op(self):
        return self

    @property
    def ops(self):
        return []

    def set_type(self, type):
        self.type = type
        return self

    def generate_add_delta(self, adjoints, delta):
        if self not in adjoints:
            adjoints[self] = delta
        else:
            adjoints[self] = delta+adjoints[self]

    def arg_shapes(self):
        return (arg.shape for arg in self.args)

    def __str__(self):
        return self.__class__.__name__

    def reshape(self, shape):
        return reshape(self, shape)

    @property
    def size(self):
        result = 1
        for d in self.shape:
            result = result*d
        return result



class OpIterValue(Op):
    def __init__(self, sequence):
        super(OpIterValue, self).__init__(sequence)

class OpIterator(Op):
    def __init__(self, sequence):
        super(OpIterator, self).__init__(sequence)
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


class input(Op):
    """
    Can be set externally.
    """

    def __init__(self, shape):
        super(input, self).__init__()
        self.shape = shape

    def evaluate(self, value):
        return value

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
        self.__op = Op.as_op(op)
        return self


class Constant(Op):
    """
    A constant that appears in a graph.
    """
    def __init__(self, const):
        super(Constant, self).__init__()
        self.const = const
        if isinstance(const, np.ndarray):
            self.shape =  const.shape
        else:
            self.shape = ()

    def evaluate(self, value):
        return self.const

    def generate_adjoints(self, tape, delta):
        pass

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.const)



def elementwise_shape(*shapes):
    n = max((len(shape) for shape in shapes))
    def prepend(s):
        return tuple(1 for x in xrange(n-len(s)))+s
    shapes = (prepend(s) for s in shapes)

    def broadcast(*vals):
        result = 1
        for val in vals:
            if val == 1:
                continue
            elif result == 1 or val == result:
                result = val
            else:
                raise IncompatibleShapesError()
        return result

    return tuple(broadcast(*d) for d in zip(*shapes))


class add(Op):
    def __init__(self, x, y, out=None):
        super(add, self).__init__(x, y)
        self.shape = elementwise_shape(*self.arg_shapes())
        self.setup_out(out)

    def evaluate(self, value, x, y):
        return x + y

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, delta)


class cos(Op):
    def __init__(self, x, out=None):
        super(cos, self).__init__(x)
        (self.shape,) = self.arg_shapes()
        self.setup_out(out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*sin(x))


# This makes the derivative simpler if we need it
def divide(x, y, out=None):
    result = x*reciprocal(y)
    result.setup_out(out)
    return result


class dot(Op):
    def __init__(self, x, y, out=None):
        xr = x
        if len(x.shape) == 0:
            xr = reshape(x, (1,1))
        elif len(x.shape) != 2:
            d1 = 1
            for d in x.shape[:-1]:
                d1 = d1*d
            xr = reshape(x, (d1, x.shape[-1]))
        yr = y
        if len(y.shape) == 0:
            yr = reshape(y, (1,1))
        elif len(y.shape) != 2:
            d1 = 1
            for d in y.shape[1:]:
                d1 = d1*d
            yr = reshape(y, (y.shape[0], d1))

        super(dot, self).__init__(xr, yr)

        xshape = x.shape
        yshape = y.shape

        if len(xshape) == 0:
            self.shape = yshape
        elif len(yshape) == 0:
            self.shape = xshape
        elif xshape[-1] != yshape[0]:
            raise IncompatibleShapesError()
        else:
            self.shape = tuple(xshape[:-1]+yshape[1:])
        self.setup_out(out)

    def evaluate(self, value, x, y):
        return np.dot(x,y,value)

    def generate_adjoints(self, adjoints, delta, x, y):

        dr = delta
        if len(dr.shape) != 2:
            dr = reshape(delta, (x.shape[0], y.shape[1]))
        x.generate_add_delta(adjoints, dot(dr, y.T))
        y.generate_add_delta(adjoints, dot(x.T, dr))


class empty(Op):
    def __init__(self, shape, dtype=None, name=None, persist_values=None, *args):
        super(empty, self).__init__()
        self.shape = shape
        self.name = name
        self.dtype = dtype
        self.persist_values = persist_values
        self.other_args = args

    def generate_adjoints(self, adjoints, delta):
        pass


class exp(Op):
    def __init__(self, x, out=None):
        super(exp, self).__init__(x)
        self.shape, _ = self.arg_shapes()
        self.setup_out(out)


    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta)


class log(Op):
    def __init__(self, x, out=None):
        super(empty, self).__init__(x)
        self.shape = elementwise_shape(*self.arg_shapes())
        self.setup_out(out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta/x)


class multiply(Op):
    def __init__(self, x, y, out=None):
        super(multiply, self).__init__(x, y)
        self.shape = elementwise_shape(*self.arg_shapes())
        self.setup_out(out)

    def evaluate(self, value, x, y):
        return np.dot(x,y,value)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta*y)
        y.generate_add_delta(adjoints, x*delta)


class negative(Op):
    def __init__(self, x, out=None):
        super(negative, self).__init__(x)
        self.shape = self.arg_shapes()[0]
        self.setup_out(out)

    def evaluate(self, value, x):
        return -x

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -delta)


class ones(Op):
    def __init__(self, shape, dtype=None, name=None, persist_values=None, *args):
        super(ones, self).__init__()
        self.shape = shape
        self.name = name
        self.dtype = dtype
        self.persist_values = persist_values
        self.other_args = args

    def generate_adjoints(self, adjoints, delta):
        pass


class reciprocal(Op):
    def __init__(self, x, out=None):
        super(reciprocal, self).__init__(x)
        self.shape, = self.arg_shapes()
        self.setup_out(out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -self*self*delta)


#TODO This should be restride, as should transpose, is terms of (i,j,k) -> ((i,j),k) i.e. remap
class reshape(Op):
    def __init__(self, x, shape):
        super(reshape, self).__init__(x)
        self.shape = shape
        if self.size != x.size:
            raise ValueError('total size of new array must be unchanged')

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, reshape(delta, x.shape))


class sig(Op):
    def __init__(self, x, out=None):
        super(sig, self).__init__(x)
        (self.shape,) = self.arg_shapes()
        self.setup_out(out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*self*(1.0-self))


class sin(Op):
    def __init__(self, x, out=None):
        super(sin, self).__init__(x)
        (self.shape,) = self.arg_shapes()
        self.setup_out(out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*cos(x))


class square(Op):
    def __init__(self, x, out=None):
        super(square, self).__init__(x)
        self.shape = self.arg_shapes()[0]
        self.setup_out(out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, 2.0*delta*x)


class subtract(Op):
    def __init__(self, x, y, out=None):
        super(subtract, self).__init__(x, y)
        self.shape = elementwise_shape(*self.arg_shapes())
        self.setup_out(out)

    def evaluate(self, value, x, y):
        return x-y

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, -delta)


class tanh(Op):
    def __init__(self, x, out=None):
        super(tanh, self).__init__(x)
        (self.shape,) = self.arg_shapes()
        self.setup_out(out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta*(1.0-self*self))


class transpose(Op):
    def __init__(self, x, out=None):
        super(transpose, self).__init__(x)
        xshape, = self.arg_shapes()
        self.shape = tuple(reversed(xshape))
        self.setup_out(out)

    def evaluate(self, value, x):
        return np.transpose(x)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta.T)


class zeros(Op):
    def __init__(self, shape, dtype=None, name=None, persist_values=None, *args):
        super(zeros, self).__init__()
        self.shape = shape
        self.name = name
        self.dtype = dtype
        self.persist_values = persist_values
        self.other_args = args

    def generate_adjoints(self, adjoints, delta):
        pass






class range(Op):
    def __init__(self, start, stop=None, step=1):
        if stop is None:
            start = 0
            stop = start
        super(range, self).__init__(start, stop, step)


class deriv(Op):
    """
    Derivative of dep with respect to indep
    """
    def __init__(self, dep, indep):
        super(deriv, self).__init__(dep, indep, Graph.get_default_graph(dep, indep).get_adjoints(dep.op)[indep.op])
        _, _, self.shape = self.arg_shapes()

    @property
    def dep(self):
        dep, indep = self.args
        return dep

    @property
    def indep(self):
        dep, indep = self.args
        return indep

    def get_computation(self, tape):
        dep, indep = self.args
        return tape.get_computation(tape.get_adjoints(dep)[indep])


class ControlBlock(object):

    def __init__(self):
        self.__ops = []


    def add_op(self, op):
        if op is not self:
            self.__ops.append(op)

    @property
    def graph(self):
        raise NotImplementedError()

    @property
    def ops(self):
        return self.__ops


class NestedControlBlock(ControlBlock, Op):
    def __init__(self, context):
        super(NestedControlBlock, self).__init__()
        self.parent_context = context

    @property
    def graph(self):
        return self.context.graph


class Iterator(NestedControlBlock):
    def __init__(self, context):
        super(Iterator, self).__init__(context)


def show_graph(g):
    ids = {}

    def opids(g):
        ids[g] = len(ids)
        for op in g.ops:
            opids(op)

    opids(g)

    def show_op(g):
        for op in g.ops:
            name = ''
            if op.name is not None:
                name = op.name
            outid = ''
            if op.out is not None:
                outid = '=>%d' % (ids[op.out],)
            print '%d:%d:%s%s:%s%s%s' % (ids[op], ids[op.context], name, op.shape, op, tuple(ids[arg] for arg in op.args), outid)
            show_op(op)
    show_op(g)

@contextmanager
def default_graph(graph=None):
    thread_data = Graph.get_thread_data()
    old_graph = thread_data.graph
    if graph is None:
        graph=Graph()
    try:
        thread_data.graph = graph
        yield(graph)
    finally:
        thread_data.graph = old_graph

class VariableBlock(object):
    def __setattr__(self, name, value):
        value.op.name = name
        super(VariableBlock, self).__setattr__(name, value)


class Graph(ControlBlock):

    def __init__(self):
        self.context = self
        super(Graph, self).__init__()
        self.inputs = {}
        self.variables = {}
        self.op_adjoints = {}

    import threading
    __thread_data = threading.local()
    __thread_data.graph = None

    @staticmethod
    def get_thread_data():
        return Graph.__thread_data

    @staticmethod
    def get_default_graph(*args):
        graph = Graph.__thread_data.graph
        for arg in args:
            if isinstance(arg, Arg):
                g = arg.graph
                if g is not None:
                    graph = g
                    break

        if graph is None:
            raise MissingGraphError()
        return graph

    @property
    def graph(self):
        return self

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
        adjoints[op] = ones(op.shape)
        for o in reversed(ordered_ops):
            o.generate_adjoints(adjoints, adjoints[o], *o.args)
        return adjoints

    @contextmanager
    def iterate(self, iterable):
        old_context = self.context
        items = iter(iterable)
        n = items.next()
        try:
            iterator = Iterator(old_context)
            self.context = iterator
            if isinstance(n, tuple):
                var = tuple(self.variable() for v in n)
                for vv, v in zip(var, n):
                    vv.set(v)
            else:
                var = self.variable()
                var.set(n)
            yield(var)
        finally:
            self.context = old_context

    # Neon backend functions
    # TODO These are just here as placeholders
    def absolute(self, a, out=None):
        pass

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

    def maximum(self, a, b, out=None):
        pass

    def mean(self, a, axis=None, partial=None, out=None, keepdims=None):
        pass

    def min(self, a, axis=None, out=None, keepdims=None):
        pass

    def minimum(self, a, b, out=None):
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

    def sqrt(self, a, out=None):
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



