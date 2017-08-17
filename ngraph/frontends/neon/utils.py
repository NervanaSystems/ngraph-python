from __future__ import absolute_import

import ngraph as ng
from .axis import ax


def make_convolution_placeholder(shape=None):
    """
    Create a placeholder op for inputs to a convolution layer

    Arguments:
        shape (tuple): The desired shape of the placeholder,
                       with axes in the order of C, D, H, W, N

    Returns:
        5-D placeholder op
    """

    H = ng.make_axis(name="H", docstring="Height")
    W = ng.make_axis(name="W", docstring="Width")
    D = ng.make_axis(name="D", docstring="Depth")
    C = ng.make_axis(name="C", docstring="Channel")

    x = ng.placeholder(axes=ng.make_axes([C, D, H, W, ax.N]))
    if shape is not None:
        x.axes.set_shape(shape)

    return x


def get_function_or_class_name(obj):

    if hasattr(obj, "__name__"):
        name = obj.__name__
    elif callable(obj):
        name = type(obj).__name__
    else:
        name = None

    return name
