from neon.util.argparser import NeonArgparser
from neon.data import ImageLoader
import geon.backends.graph.dataloaderbackend

import geon.backends.graph.funs as be
import geon.backends.graph.graph as graph
import geon.backends.graph.evaluation as evaluation
import numpy as np

class MyRnn(be.Model):

    def __init__(self, **kargs):
        super(MyRnn, self).__init__(**kargs)
        # g: graph node root namespace
        # a: axis namespace
        # v: symbolic variable namespace
        g = self.graph

        # Define the axes
        g.N = be.axis()
        g.T = be.axis(dependent=g.N)
        g.X = be.axis()
        g.Y = be.axis()
        g.H = be.axis()

        # Define the inputs.
        g.x = be.input(axes=(g.X, g.T, g.N))
        # This would only be used for training or evaluation
        g.y_ = be.input(axes=(g.Y, g.T, g.N))

        # Recursive computation of the hidden state.
        # Axes for defining position roles
        h = be.recurse(axes=(g.H, g.T, g.N))
        h[:, 0] = be.parameter(axes=(g.H))
        HWh = be.parameter(axes=(g.H, g.H))
        HWx = be.parameter(axes=(g.X, g.H))
        Hb = be.parameter(axes=(g.H,))

        g.t = be.variable(type=g.T)
        h[:, g.t+1] = be.sig(be.dot(h[:, g.t], HWh)+be.dot(g.x[g.T], HWx)+Hb)

        YW = be.parameter(axes=(g.H, g.Y))
        Yb = be.parameter(axes=(g.Y))
        # This is what we would want for inference
        g.y = be.tanh(be.dot(h, YW)+Yb)

        e = g.y-g.y_
        # This is what we want for training, perhaps added to a parameter regularization
        g.error = be.dot(e, e)/len(e)

