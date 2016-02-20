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

class Computation(object):
    """Slightly 'compiled' version of something to be autodiff'd.

    """
    class Function(object):
        """
        One step of a computation.
        """
        def __init__(self, value, id, child_ids):
            self.value = value
            self.id = id
            self.child_ids = child_ids

    def __init__(self, value):
        ids = {}
        self.functions = []
        self.inputs = {}
        self.variables = {}
        self.size = 0

        def get_id(value):
            if value in ids:
                return ids[value]
            id = self.size
            self.size += 1
            ids[value] = id
            return id

        def add_function(value):
            id = get_id(value)
            function = Computation.Function(value, id, tuple(ids[v] for v in value.inputs))
            self.functions.append(function)
            return function

        def add_variable(value):
            if isinstance(value, Variable):
                function = add_function(value)
                self.variables[value] = function

        def add_input(value):
            if isinstance(value, Input):
                function = add_function(value)
                self.inputs[value] = function

        def depth_first(value, f):
            if value in ids:
                return
            for v in value.inputs:
                depth_first(v, f)
            f(value)

        # Perform a topological sort on the computation steps

        # Find the variables and inputs first, to make it easier to set their values
        depth_first(value, add_variable)
        depth_first(value, add_input)

        # Now pick up the remaining
        depth_first(value, add_function)

        self.output = self.functions[get_id(value)]



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

    def compute(self, value, *inputs):
        """
        Compute a value from inputs.

        :param value: Previously computed value
        :param inputs: Values computed from previous steps
        :return: the result, typically value
        """
        raise NotImplementedError()

    def diff(self, value, value_bar, *input_values):
        """
        Push derivative deltas back

        :param value: Value computed by this step
        :param value_bar: Previously computed deriv, for resuse
        :param input_values: Values computed by inputs
        :return: deriv delta for each input
        """
        raise NotImplementedError()

    def update(self, value, value_bar, e):
        # Update the value by the deriv weighted by e
        return value

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


class Input(Value):
    """
    An input to a computation.
    """
    def __init__(self):
        super(Input, self).__init__()

    def compute(self, value):
        return value

    def diff(self, value, value_bar):
        return ()


class Variable(Value):
    """
    A variable whose autodiff will be computed.
    """
    def __init__(self):
        super(Variable, self).__init__()

    def compute(self, value):
        return value

    def diff(self, value, value_bar):
        return ()

    def update(self, value, value_bar, e):
        np.multiply(value_bar, e, value_bar)
        return np.add(value, value_bar, value)



class Constant(Value):
    """
    A constant that appears in a computation.
    """
    def __init__(self, const):
        super(Constant, self).__init__()
        self.const = const

    def compute(self, value):
        return self.const

    def diff(self, value, value_bar):
        return ()


class Neg(Value):
    def __init__(self, x):
        super(Neg, self).__init__((x,))

    def compute(self, value, x):
        return -x

    def diff(self, value, value_bar, x):
        return -value_bar


class Add(Value):
    def __init__(self, x, y):
        super(Add, self).__init__((x, y))

    def compute(self, value, x, y):
        return x + y

    def diff(self, value, value_bar, x, y):
        return (value_bar, value_bar)


class Sub(Value):
    def __init__(self, x, y):
        super(Sub, self).__init__((x,y))

    def compute(self, value, x, y):
        return x-y

    def diff(self, value, value_bar, x, y):
        return (value_bar, -value_bar)


class Mul(Value):
    def __init__(self, x, y):
        super(Mul, self).__init__((x,y))

    def compute(self, value, x, y):
        return np.dot(x,y,value)

    def diff(self, value, value_bar, x, y):
        return (np.dot(value_bar, y.T),
                np.dot(x.T, value_bar))

class Transpose(Value):
    def __init__(self, x):
        super(Transpose, self).__init__((x,))

    def compute(self, value, x):
        return np.transpose(x)

    def diff(self, value, value_bar, x):
        return value_bar.T


class Context(object):
    class Frame(object):
        def __init__(self, context, function):
            self.function = function
            self.value = None
            self.value_bar = None
            self.input_frames = tuple(context.frames[cid] for cid in function.child_ids)

        def compute(self):
            self.value = self.function.value.compute(self.value, *(tuple(frame.value for frame in self.input_frames)))

        def diff(self):
            for frame, delta in zip(self.input_frames, self.function.value.diff(self.value, self.value_bar, *tuple(frame.value for frame in self.input_frames))):
                if frame.value_bar is None:
                    frame.value_bar = delta
                else:
                    frame.value_bar += delta

    def __init__(self, computation):
        self.computation = computation
        self.frames = [None]*computation.size
        for function in computation.functions:
            self.frames[function.id] = Context.Frame(self, function)
        self.output = self.frames[computation.output.id]


    def set_input(self, input, value):
        input_function = self.computation.inputs[input]
        self.frames[input_function.id].value = value

    def init_variable(self, variable, value):
        variable_function = self.computation.variables[variable]
        self.frames[variable_function.id].value = value

    def get_variable_deriv(self, variable):
        variable_function = self.computation.variables[variable]
        return self.frames[variable_function.id].value_bar

    def get_variable_value(self, variable):
        variable_function = self.computation.variables[variable]
        return self.frames[variable_function.id].value

    def execute(self):
        for frame in self.frames:
            frame.compute()
        self.output.value_bar = np.array([1.0]).reshape(1,1)
        for frame in reversed(self.frames):
            frame.diff()

        return self.output.value

    def update(self, e):
        for frame in self.frames:
            frame.value = frame.function.value.update(frame.value, frame.value_bar, e)
            frame.value_var = None


def norm2(x):
    return x.T*x


def f():
    # Variables will be computed
    w = Variable()
    b = Variable()

    # Inputs for the function
    x = Input()
    y0 = Input()

    y = w*x+b
    e = norm2(y-y0)

    # Set up the computation
    computation = Computation(e)

    # Prepare to run the computation
    context = Context(computation)
    context.init_variable(w, np.zeros((3,4)))
    context.init_variable(b, np.zeros((3,1)))

    # For now, only one x,y pair of values
    context.set_input(x, np.array([1,1,2,1]).reshape((4,1)))
    context.set_input(y0, np.array([2,1,2]).reshape(3,1))

    for i in range(1000):
        result = context.execute()
        db = context.get_variable_deriv(b)
        dw = context.get_variable_deriv(w)
        print('e=%s dw=%s db=%s' % (result, dw, db))
        context.update(-.1/(1.0+i))

    print('w')
    print(context.get_variable_value(w))
    print('b')
    print(context.get_variable_value(b))

f()


