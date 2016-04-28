import geon.backends.graph.graph as graph

from geon.backends.graph.names import VariableBlock, AxisGenerator, name_context, layers_named, with_name_context
from geon.backends.graph.graph import Model, Parameter, deriv, input, axes_list
from geon.backends.graph.environment import bound_environment

from geon.backends.graph.graph import absolute, add, cos, divide, dot, empty, exp, log, maximum, minimum, multiply
from geon.backends.graph.graph import negative, ones, reciprocal, reshape, sig, sin, sqrt, square, subtract
from geon.backends.graph.graph import tanh, transpose, zeros, range, iterate


def relu(x,out):
    maximum(x, 0, out)

iterate = graph.iterate




