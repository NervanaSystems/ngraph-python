import numpy as np

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

    def __repr__(self):
        '%s%s' % (self.__class__, self.inputs)

    def __str__(self):
        return self.__repr__()


    # TODO This will take a backend compiler as an argument and use it to do the appropriate thing.
    # TODO Probably there are several methods that replace this, some which optimize the graph
    # TODO and some which generate/execute, with the backend compiler driving the prcoess.
    def compute(self, value, *inputs):
        """
        Compute a value from inputs.

        :param value: Previously computed value
        :param inputs: Values computed from previous steps
        :return: the result, typically value
        """
        raise NotImplementedError()

    # TODO: These should generate new nodes.  To do this we need something that associates these nodes with
    # TODO: their derivatives nodes, since each diff will add a new sum node.  The result will be a computation
    # TODO: graph that also computes the derivative.
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

#TODO empty()
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


#TODO zeros(), ones()
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
    """
    Elementwise negation.
    """
    def __init__(self, x):
        super(Neg, self).__init__((x,))

    def compute(self, value, x):
        return -x

    def diff(self, value, value_bar, x):
        return -value_bar


class Add(Value):
    """
    Elementwise addition.
    """
    def __init__(self, x, y):
        super(Add, self).__init__((x, y))

    def compute(self, value, x, y):
        return x + y

    def diff(self, value, value_bar, x, y):
        return (value_bar, value_bar)


class Sub(Value):
    """
    Elementwise subtraction.
    """
    def __init__(self, x, y):
        super(Sub, self).__init__((x,y))

    def compute(self, value, x, y):
        return x-y

    def diff(self, value, value_bar, x, y):
        return (value_bar, -value_bar)


class Mul(Value):
    """
    Elementwise multiplication.
    """
    def __init__(self, x, y):
        super(Mul, self).__init__((x,y))

    def compute(self, value, x, y):
        return np.mul(x,y,value)

    def diff(self, value, value_bar, x, y):
        raise NotImplementedError()


#TODO Elementwise division

#TODO Allow axes to be specified
#TODO dot(A,B).axis(4) or dot(A,B, axis=4)
class Dot(Value):
    """
    Dot product.
    """
    def __init__(self, x, y):
        super(Mul, self).__init__((x,y))

    def compute(self, value, x, y):
        return np.dot(x,y,value)

    def diff(self, value, value_bar, x, y):
        return (np.dot(value_bar, y.T),
                np.dot(x.T, value_bar))


#TODO Allow general dim shuffle
#TODO A.T.shuffle(2,1,4,3)
class Transpose(Value):
    """
    Transposition.
    """
    def __init__(self, x):
        super(Transpose, self).__init__((x,))

    def compute(self, value, x):
        return np.transpose(x)

    def diff(self, value, value_bar, x):
        return value_bar.T

#TODO Assign: Two cases
# 1) Capturing a computed value
# 2) Overwriting a value -- anything that uses its "out" arg

#TODO Random number generator/seed