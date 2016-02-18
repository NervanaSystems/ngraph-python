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

def as_value(x):
    if isinstance(x, Value):
        return x
    return Constant(x)


class Value(object):
    """Wraps a value do that a computation can be captured.
        """

    def __init__(self, inputs=()):
        self.inputs = tuple(as_value(input) for input in inputs)
        self.id = None

    # The computed value
    def value(self, values):
        return values[self.id]

    # The derivative
    def dvalue(self, deltas):
        return deltas[self.id]

    def tape(self, tape):
        if None == self.id:
            for input in self.inputs:
                input.tape(tape)
            self.id = len(tape)
            tape.append(self)

    def reset(self):
        self.id = None

    def input_values(self, values):
        return tuple(values[input.id] for input in self.inputs)

    def add_delta(self, deltas, delta):
        deltas[self.id] += delta

    def diff(self, values, deltas, delta):
        raise NotImplementedError()

    def dodiff(self):
        self.reset()
        tape = []
        self.tape(tape)
        values = [0]*len(tape)
        for value in tape:
            values[value.id] = value.compute(values)
        deltas = [0]*len(tape)
        deltas[self.id] = 1.0
        for value in reversed(tape):
            value.diff(values, deltas, deltas[value.id])
        return values, deltas

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
        return Mul(Node.node(val), self)

    def __div__(self, val):
        return Div(self, val)

    def __rdiv__(self, val):
        return Div(val, self)

    def __pow__(self, val):
        return Pow(self, val)

    def __rpow__(self, val):
        return Pow(self, val)


class Input(Value):
    def __init__(self, input):
        super(Input, self).__init__()
        self.input = input

    def compute(self, values):
        return self.input

    def diff(self, values, deltas, delta):
        pass


class Variable(Value):
    def __init__(self, var_value):
        super(Variable, self).__init__()
        self.var_value = var_value

    def compute(self, values):
        return self.var_value

    def diff(self, values, deltas, delta):
        pass


class Constant(Value):
    def _init__(self, const):
        super(Constant, self).__init__()
        self.const = const

    def compute(self, values):
        return self.const

    def diff(self, values, deltas, delta):
        pass


class Neg(Value):
    def __init__(self, x):
        super(Neg, self).__init__((x,))

    def compute(self, values):
        x = self.input_values(values)
        return -x

    def diff(self, values, deltas, value_bar):
        x = self.inputs
        x.add_delta(deltas, -value_bar)


class Add(Value):
    def __init__(self, x, y):
        super(Add, self).__init__((x, y))

    def compute(self, values):
        x, y = self.input_values(values)
        return x+y

    def diff(self, values, deltas, value_bar):
        x, y = self.inputs
        x.add_delta(deltas, value_bar)
        y.add_delta(deltas, value_bar)


class Sub(Value):
    def __init__(self, x, y):
        super(Sub, self).__init__((x,y))

    def compute(self, values):
        x, y = self.input_values(values)
        return x-y

    def diff(self, values, deltas, value_bar):
        x, y = self.inputs
        x.add_delta(deltas, value_bar)
        y.add_delta(deltas, -value_bar)


class Mul(Value):
    def __init__(self, x, y):
        super(Mul, self).__init__((x,y))

    def compute(self, values):
        x, y = self.input_values(values)
        return x*y

    def diff(self, values, deltas, value_bar):
        x, y = self.inputs
        xv, yv = self.input_values(values)
        x.add_delta(deltas, value_bar*yv)
        y.add_delta(deltas, value_bar*xv)

def f():
    w = Variable(2)
    b = Variable(1)
    x = Input(4)
    y = w*x+b
    values, deltas = y.dodiff()
    print(y.value(values))
    print(w.dvalue(deltas))
    print(b.dvalue(deltas))

f()


