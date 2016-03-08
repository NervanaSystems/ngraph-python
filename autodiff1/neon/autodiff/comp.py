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

# TBD: Generate computation rather than executing
# TBD: Can we take some ideas from autograd?
# TBD: Conditionals, convolution, etc.
# TBD: Sanitize shapes, deriv conventions


import numpy as np


class Tape(object):
    """Slightly 'compiled' version of something to be autodiff'd.

    This is the "tape"

    """
    class Computation(object):
        """
        One step of a computation.
        """
        def __init__(self, value, id, child_ids):
            self.value = value
            self.id = id
            self.child_ids = child_ids

    def __init__(self, *values):
        """

        :param value: The value to compute
        :return:
        """

        self.ids = {}
        self.computations = []
        self.value_adjoints = {}

        # Perform a topological sort on the computation steps
        for value in values:
            self.add_computation(value)

    def __len__(self):
        return len(self.computations)

    def add_computation(self, value):
        if value in self.ids:
            return self.ids[value]

        if isinstance(value, Deriv):
            self.add_computation(value.dep)
            self.add_computation(value.indep)
            adjoints = self.get_adjoints(value.dep)
            variable = value.indep
            computation = self.add_computation(adjoints[variable])
            self.ids[value] = computation
            return computation

        for v in value.inputs:
            self.add_computation(v)

        id = len(self.computations)
        computation = Tape.Computation(value, id, tuple(self.ids[v].id for v in value.inputs))
        self.ids[value] = computation
        self.computations.append(computation)
        return computation

    def get_adjoints(self, value):
        if value in self.value_adjoints:
            return self.value_adjoints[value]
        adjoints = {}
        self.value_adjoints[value] = adjoints
        adjoints[value] = Constant(np.array([1.0]).reshape(1, 1))
        for computation in reversed(self.computations):
            value = computation.value
            value.generate_adjoints(adjoints, adjoints[value], *value.inputs)
        return adjoints


class Value(object):
    """
    A combination of python magic method handler and vm for computations.
    """

    @staticmethod
    def as_value(x):
        if isinstance(x, Value):
            return x
        return Constant(x)


    def __init__(self, inputs=()):
        self.inputs = tuple(Value.as_value(input) for input in inputs)

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


class Deriv(Value):
    def __init__(self, dep, indep):
        super(Deriv, self).__init__()
        self.dep = dep
        self.indep = indep

    def evaluate(self, value):
        return value


class Input(Value):
    """
    An input to a computation.
    """
    def __init__(self):
        super(Input, self).__init__()

    def evaluate(self, value):
        return value

    def generate_adjoints(self, tape, delta):
        pass


class Variable(Value):
    """
    A variable whose autodiff will be computed.
    """
    def __init__(self):
        super(Variable, self).__init__()

    def evaluate(self, value):
        return value

    def generate_adjoints(self, tape, delta):
        pass


class Constant(Value):
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

class Neg(Value):
    def __init__(self, x):
        super(Neg, self).__init__((x,))

    def evaluate(self, value, x):
        return -x

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -delta)


class Add(Value):
    def __init__(self, x, y):
        super(Add, self).__init__((x, y))

    def evaluate(self, value, x, y):
        return x + y

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, delta)



class Sub(Value):
    def __init__(self, x, y):
        super(Sub, self).__init__((x,y))

    def evaluate(self, value, x, y):
        return x-y

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, -delta)


class Mul(Value):
    def __init__(self, x, y):
        super(Mul, self).__init__((x,y))

    def evaluate(self, value, x, y):
        return np.dot(x,y,value)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, np.dot(delta, y.T))
        y.generate_add_delta(adjoints, np.dot(x.T, delta))


class Transpose(Value):
    def __init__(self, x):
        super(Transpose, self).__init__((x,))

    def evaluate(self, value, x):
        return np.transpose(x)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta.T)


class Context(object):

    def __init__(self, tape):
        self.tape = tape
        self.values = [None] * len(tape)

    def set(self, variable, value):
        self.values[self.tape.ids[variable].id] = value

    def get(self, variable):
        return self.values[self.tape.ids[variable].id]

    def compute(self):
        for computation in self.tape.computations:
            id = computation.id
            self.values[id] = computation.value.evaluate(self.values[id], *(tuple(self.values[child_id] for child_id in computation.child_ids)))

    def get_deriv(self, variable_map, variable):
        adjoint_id, id = variable_map[variable]
        return self.values[adjoint_id]


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

        context.compute()
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


