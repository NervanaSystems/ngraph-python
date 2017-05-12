from __future__ import absolute_import
import ngraph as ng
from .axis import ax


def make_convolution_placeholder(shape=None):

    H = ng.make_axis(name="H", docstring="Height")
    W = ng.make_axis(name="W", docstring="Width")
    D = ng.make_axis(name="D", docstring="Depth")
    C = ng.make_axis(name="C", docstring="Channel")

    x = ng.placeholder(axes=ng.make_axes([C, D, H, W, ax.N]))
    if shape is not None:
        x.axes.set_shape(shape)

    return x
