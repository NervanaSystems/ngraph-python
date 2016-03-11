
import numpy as np
from geon.backends.graph.graph import Graph

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

    be = Graph()
    w = be.input('w')
    b = be.input('b')

    xs = be.input('xs')
    y0s = be.input('y0s')

    with be.iterate(be.range(be.input('N'))) as i:
        e = be.variable('e').set(0)
        with be.iterate(zip(xs, y0s)) as ((x, y0)):
            e.set(e + norm2(w * x + b))

        dedw = be.deriv(e,w)
        dedb = be.deriv(e,b)

        alpha = -1.0/(1.0+i)

        w.set(w+alpha*dedw)
        b.set(b+alpha*dedb)

f()
