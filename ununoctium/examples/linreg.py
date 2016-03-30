
import numpy as np
import geon.backends.graph.graph as graph
import geon.backends.graph.typing as gt

def generate_samples(xsize, bsize, N):
    w = np.random.rand(bsize, xsize)
    b = np.random.rand(bsize)
    xs = np.empty((xsize, N))
    ys = np.empty((bsize, N))
    for i in range(N):
        x = np.random.rand(xsize)
        y = np.dot(w,x)+b
        xs[:,i] = x
        ys[:,i] = y
    return xs, ys, w, b


def norm2(x):
    return x.T*x


def f():
    xsize = 10
    bsize = 3
    N = 100
    K = 30

    from geon.backends.graph.funs import *

    gr = graph.Graph()
    with graph.default_graph(gr) as g:

        g.w = zeros((xsize, bsize))
        g.b = zeros((bsize,))

        g.xs = input((xsize, N))
        g.y0s = input((bsize, N))

        with iterate(range(K)) as g.i:
            g.e = 0
            with iterate(range(N)) as g.n:
                g.x = g.xs[:,g.n]
                g.y0 = g.y0s[:,g.n]
                g.e = g.e + norm2(g.w * g.x + g.b)

            g.dedw = deriv(g.e, g.w)
            g.dedb = deriv(g.e, g.b)

            g.alpha = -1.0/(1.0+g.i)

            g.w += g.alpha*g.dedw
            g.b += g.alpha*g.dedb

    graph.show_graph(gr)

f()
