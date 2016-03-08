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

# First cut based on http://www.qucosa.de/fileadmin/data/qucosa/documents/827/1206719130404-2230.pdf

# TBD: Can we take some ideas from autograd?
# TBD: Conditionals, convolution, etc.

import numpy as np


class Tape(object):
    """Slightly 'compiled' version of a computation.

    This is the "tape"
    """
    class Computation(object):
        """
        One step of a computation.
        """
        def __init__(self, op, pos, pos_children):
            self.op = op
            self.pos = pos
            self.pos_children = pos_children

        def evaluate(self, values):
            values[self.pos] = self.op.evaluate(values[self.pos], *(tuple(values[pos_child] for pos_child in self.pos_children)))

        def get(self, values):
            return values[self.pos]

        def set(self, values, value):
            values[self.pos] = value


    def __init__(self, *ops):
        """
        :param ops: The values to compute
        :return:
        """

        self.op_computations = {}
        self.computations = []
        self.op_adjoints = {}

        # Perform a topological sort on the computation steps
        for op in ops:
            self.get_computation(op)

    def __len__(self):
        return len(self.computations)

    def new_computation(self, op):
        pos = len(self.computations)
        computation = Tape.Computation(op, pos, tuple(self.op_computations[child].pos for child in op.children))
        self.computations.append(computation)
        return computation

    def get_computation(self, op):
        if op in self.op_computations:
            return self.op_computations[op]

        for child in op.children:
            self.get_computation(child)

        computation = op.get_computation(self)
        self.op_computations[op] = computation
        return computation

    def get_adjoints(self, op):
        if op in self.op_adjoints:
            return self.op_adjoints[op]
        adjoints = {}
        self.op_adjoints[op] = adjoints
        adjoints[op] = Constant(np.array([1.0]).reshape(1, 1))
        for computation in reversed(self.computations):
            op = computation.op
            op.generate_adjoints(adjoints, adjoints[op], *op.children)
        return adjoints

class Context(object):
    """
    An execution context for evaluating a computation on a tape.
    """
    def __init__(self, tape):
        self.tape = tape
        self.values = [None] * len(tape)

    def set(self, op, value):
        self.tape.op_computations[op].set(self.values, value)

    def get(self, op):
        return self.tape.op_computations[op].get(self.values)

    def evaluate(self):
        for computation in self.tape.computations:
            computation.evaluate(self.values)


class Op(object):
    """
    A node in a compute graph.
    """

    @staticmethod
    def os_op(x):
        if isinstance(x, Op):
            return x
        return Constant(x)

    def __init__(self, children=()):
        self.children = tuple(Op.os_op(child) for child in children)

    def get_computation(self, tape):
        return tape.new_computation(self)

    def evaluate(self, value, *inputs):
        """
        Compute a value from inputs.

        :param value: Previously computed value
        :param inputs: Values computed from previous steps
        :return: the result, typically value
        """
        raise NotImplementedError()

    def generate_adjoints(self, adjoints, delta, *input_values):
        raise NotImplementedError()

    def generate_add_delta(self, adjoints, delta):
        if self not in adjoints:
            adjoints[self] = delta
        else:
            adjoints[self] = delta+adjoints[self]

    # Magic methods for builtin operations we want to use for creating nodes
    def __neg__(self):
        return Neg(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return Abs(self)

    def __add__(self, val):
        return Add(self, val)

    def __radd__(self, val):
        return Add(val, self)

    def __sub__(self, val):
        return Sub(self, val)

    def __rsub__(self, val):
        return Sub(val, self)

    def __mul__(self, val):
        return Mul(self, val)

    def __rmul__(self, val):
        return Mul(val, self)

    def __div__(self, val):
        return Div(self, val)

    def __rdiv__(self, val):
        return Div(val, self)

    def __pow__(self, val):
        return Pow(self, val)

    def __rpow__(self, val):
        return Pow(self, val)

    @property
    def T(self):
        return Transpose(self)


class Deriv(Op):
    """
    Derivative of dep with respect to indep
    """
    def __init__(self, dep, indep):
        super(Deriv, self).__init__((dep, indep))

    @property
    def dep(self):
        dep, indep = self.children
        return dep

    @property
    def dep(self):
        dep, indep = self.children
        return indep

    def get_computation(self, tape):
        dep, indep = self.children
        return tape.get_computation(tape.get_adjoints(dep)[indep])


class Input(Op):
    """
    An input to a computation.
    """
    def __init__(self):
        super(Input, self).__init__()

    def evaluate(self, value):
        return value

    def generate_adjoints(self, tape, delta):
        pass


class Variable(Op):
    """
    A variable whose autodiff will be computed.
    """
    def __init__(self):
        super(Variable, self).__init__()

    def evaluate(self, value):
        return value

    def generate_adjoints(self, tape, delta):
        pass


class Constant(Op):
    """
    A constant that appears in a computation.
    """
    def __init__(self, const):
        super(Constant, self).__init__()
        self.const = const

    def evaluate(self, value):
        return self.const

    def generate_adjoints(self, tape, delta):
        pass

class Neg(Op):
    def __init__(self, x):
        super(Neg, self).__init__((x,))

    def evaluate(self, value, x):
        return -x

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -delta)


class Add(Op):
    def __init__(self, x, y):
        super(Add, self).__init__((x, y))

    def evaluate(self, value, x, y):
        return x + y

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, delta)



class Sub(Op):
    def __init__(self, x, y):
        super(Sub, self).__init__((x,y))

    def evaluate(self, value, x, y):
        return x-y

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, -delta)


class Mul(Op):
    def __init__(self, x, y):
        super(Mul, self).__init__((x,y))

    def evaluate(self, value, x, y):
        return np.dot(x,y,value)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, np.dot(delta, y.T))
        y.generate_add_delta(adjoints, np.dot(x.T, delta))


class Transpose(Op):
    def __init__(self, x):
        super(Transpose, self).__init__((x,))

    def evaluate(self, value, x):
        return np.transpose(x)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta.T)


def norm2(x):
    return x.T*x


def f():
    adiff = True

    # Variables will be computed
    w = Variable()
    b = Variable()

    # Inputs for the function
    x = Input()
    y0 = Input()

    y = w*x+b
    e = norm2(y-y0)

    dedw = Deriv(e,w)
    dedb = Deriv(e,b)

    # Set up the computation

    tape = Tape(e, dedw, dedb)

    # Prepare to run the computation
    context = Context(tape)
    context.set(w, np.zeros((3,4)))
    context.set(b, np.zeros((3,1)))


    for i in range(1000):
    # For now, only one x,y pair of values
        context.set(x, np.array([1, 1, 2, 1]).reshape((4, 1)))
        context.set(y0, np.array([2, 1, 2]).reshape(3, 1))

        context.evaluate()
        result = context.get(e)
        db = context.get(dedb)
        dw = context.get(dedw)

        print('e=%s dw=%s db=%s' % (result, dw, db))
        context.set(w, context.get(w)-.1/(1.0+i)*dw)
        context.set(b, context.get(b)-.1/(1.0+i)*db)

    print('w')
    print(context.get(w))
    print('b')
    print(context.get(b))

f()


