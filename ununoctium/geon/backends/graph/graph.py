from contextlib import contextmanager

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

class Arg(object):
    """
    An Arg is something that can appear as an argument; variables and computation results.
    """

    @property
    def op(self):
        raise NotImplementedError()

    @property
    def graph(self):
        raise NotImplementedError()

    def __iter__(self):
        return OpIterator(self.graph, self.op)

    # Magic methods for builtin operations we want to use for creating nodes
    def __neg__(self):
        return Neg(self.graph, self)

    def __pos__(self):
        return self

    def __abs__(self):
        return Abs(self.graph, self)

    def __add__(self, val):
        return Add(self.graph, self, val)

    def __radd__(self, val):
        return Add(self.graph, val, self)

    def __sub__(self, val):
        return Sub(self.graph, self, val)

    def __rsub__(self, val):
        return Sub(self.graph, val, self)

    def __mul__(self, val):
        return Mul(self.graph, self, val)

    def __rmul__(self, val):
        return Mul(self.graph, val, self)

    def __div__(self, val):
        return Div(self.graph, self, val)

    def __rdiv__(self, val):
        return Div(self.graph, val, self)

    def __pow__(self, val):
        return Pow(self.graph, self, val)

    def __rpow__(self, val):
        return Pow(self.graph, self, val)

    @property
    def T(self):
        return Transpose(self.graph, self)


class Op(Arg):
    """
    An Op is the result of some sort of operation.
    """
    def __init__(self, graph, *args):
        self.context = graph.context
        self.args = tuple(Op.as_op(graph, arg) for arg in args)
        self.context.add_op(self)

    @staticmethod
    def as_op(graph, x):
        if isinstance(x, Arg):
            return x.op

        return Constant(graph, x)

    @property
    def graph(self):
        return self.context.graph

    @property
    def op(self):
        return self


class OpIterValue(Op):
    def __init__(self, graph, sequence):
        super(OpIterValue, self).__init__(graph, sequence)

class OpIterator(Op):
    def __init__(self, graph, sequence):
        super(OpIterator, self).__init__(graph, sequence)
        self.__next = True

    def __iter__(self):
        return OpIterator(self.graph, *self.args)

    def next(self):
        if self.__next:
            self.__next = False
            sequence, = self.args
            return OpIterValue(self.graph, sequence)
        else:
            raise StopIteration()


class Input(Op):
    """
    Can be set externally.
    """

    def __init__(self, graph, name):
        super(Input, self).__init__(graph)
        self.name = name

    def evaluate(self, value):
        return value

    def generate_adjoints(self, tape, delta):
        pass


class Constant(Op):
    """
    A constant that appears in a graph.
    """
    def __init__(self, graph, const):
        super(Constant, self).__init__(graph)
        self.const = const

    def evaluate(self, value):
        return self.const

    def generate_adjoints(self, tape, delta):
        pass


class Neg(Op):
    def __init__(self, graph, x):
        super(Neg, self).__init__(graph, x)

    def evaluate(self, value, x):
        return -x

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -delta)


class Add(Op):
    def __init__(self, graph, x, y):
        super(Add, self).__init__(graph, x, y)

    def evaluate(self, value, x, y):
        return x + y

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, delta)



class Sub(Op):
    def __init__(self, graph, x, y):
        super(Sub, self).__init__(graph, x, y)

    def evaluate(self, value, x, y):
        return x-y

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, -delta)


class Mul(Op):
    def __init__(self, graph, x, y):
        super(Mul, self).__init__(graph, x, y)

    def evaluate(self, value, x, y):
        return np.dot(x,y,value)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, np.dot(delta, y.T))
        y.generate_add_delta(adjoints, np.dot(x.T, delta))


class Div(Op):
    def __init__(self, graph, x, y):
        super(Div, self).__init__(graph, x, y)




class Transpose(Op):
    def __init__(self, graph, x):
        super(Transpose, self).__init__(graph, x)

    def evaluate(self, value, x):
        return np.transpose(x)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta.T)


class Deriv(Op):
    """
    Derivative of dep with respect to indep
    """
    def __init__(self, graph, dep, indep):
        super(Deriv, self).__init__(graph, dep, indep)

    @property
    def dep(self):
        dep, indep = self.children
        return dep

    @property
    def indep(self):
        dep, indep = self.children
        return indep

    def get_computation(self, tape):
        dep, indep = self.children
        return tape.get_computation(tape.get_adjoints(dep)[indep])


class Variable(Arg):

    def __init__(self, graph, name=None):
        self.__graph = graph
        self.name = name
        self.__op = None

    @property
    def graph(self):
        return self.__graph

    @property
    def op(self):
        if self.__op is None:
            raise UnititializedVariableError()
        return self.__op

    def set(self, op):
        self.__op = Op.as_op(self.graph, op)
        return self


class Range(Op):
    def __init__(self, graph, start, stop=None, step=1):
        if stop is None:
            start = 0
            stop = start
        super(Range, self).__init__(graph, start, stop, step)



class ControlBlock(object):

    def __init__(self):
        self.ops = []
        self.contexts = []

    def add_op(self, op):
        self.ops.append(op)

    def add_context(self, context):
        self.contexts.append(context)

    @property
    def graph(self):
        return NotImplementedError()

class RootControlBlock(ControlBlock):
    def __init__(self, graph):
        super(RootControlBlock, self).__init__()
        self.__graph = graph

    @property
    def graph(self):
        return self.__graph

class NestedControlBlock(ControlBlock):
    def __init__(self, context):
        super(NestedControlBlock, self).__init__()
        self.context = context
        self.context.add_context(self)

    @property
    def graph(self):
        return self.context.graph


class Iterator(NestedControlBlock):
    def __init__(self, context):
        super(Iterator, self).__init__(context)


class Graph(object):

    def __init__(self):
        self.root_context = RootControlBlock(self)
        self.context = self.root_context
        self.inputs = {}
        self.variables = {}

    def input(self, name):
        """
        A variable whose initial value can be set in a graph.
        :param name:
        :return:
        """
        input = Input(self, name)
        self.inputs[name] = input
        variable = self.variable(name)
        variable.set(input)
        return variable

    def variable(self, name=None):
        variable = Variable(self, name)
        self.variables[name] = variable
        return variable

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

    def range(self, *args):
        return Range(self, *args)

    def deriv(self, dep, indep):
        return Deriv(self, dep, indep)

    # Neon backend
    def absolute(self, a, out=None):
        pass

    def add(self, a, b, out=None):
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

    def divide(self, a, b, out=None):
        pass

    def dot(self, a, b, out=None):
        pass

    def empty(self, shape, dtype=None, name=None, persist_values=None, *args):
        pass

    def empty_like(self, other_ary, name=None, persist_values=None):
        pass

    def end(self, block, identifier):
        pass

    def equal(self, a, b, out=None):
        pass

    def exp(self, a, out=None):
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

    def log(self, a, out=None):
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

    def multiply(self, a, b, out=None):
        pass

    def negative(self, a, out=None):
        pass

    def not_equal(self, a, b, out=None):
        pass

    def onehot(self, indices, axis, out=None):
        pass

    def ones(self, shape, dtype=None, name=None, persist_values=None, *args):
        pass

    def output_dim(self, X, S, padding, strides, pooling=None):
        pass

    def pool_layer(self, dtype, op, N, C, D=None, H=None, W=None, J=None, T=None, *args):
        pass

    def power(self, a, b, out=None):
        pass

    def reciprocal(self, a, out=None):
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

    def sig(self, a, out=None):
        pass

    def sig2(self, a, out=None):
        pass

    def sqrt(self, a, out=None):
        pass

    def square(self, a, out=None):
        pass

    def std(self, a, axis=None, partial=None, out=None, keepdims=None):
        pass

    def subtract(self, a, b, out=None):
        pass

    def sum(self, a, axis=None, out=None, keepdims=None):
        pass

    def take(self, a, indices, axis, out=None):
        pass

    def tanh(self, a, out=None):
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

    def zeros(self, shape, dtype=None, name=None, persist_values=None, *args):
        pass

    def zeros_like(self, other_ary, name=None, persist_values=None):
        pass








