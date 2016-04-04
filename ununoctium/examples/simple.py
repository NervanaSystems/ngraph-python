import geon.backends.graph.graph as graph
import geon.backends.graph.evaluation as evaluation
import numpy as np

gr = graph.Graph()
with graph.default_graph(gr) as g:
    import geon.backends.graph.funs as be

    def norm2(x):
        return be.dot(x.T,x)
    try:
        g.w = be.zeros((3, 10))
        g.b = be.zeros((3,))

        g.x = be.input((10,))
        g.y0 = be.input((3,))

        g.e = norm2(be.sig(be.dot(g.w, g.x) + g.b))
        g.reg = norm2(g.b) + norm2(be.reshape(g.w, (g.w.size,)))
        g.l = g.e + .1 * g.reg

        g.dedw = be.deriv(g.l, g.w)
        g.dedb = be.deriv(g.l, g.b)

    except graph.IncompatibleShapesError:
        # Convenient place to put a breakpoint for debugging
        pass

#graph.show_graph(gr)

gnp = evaluation.GenNumPy(graph=gr)
gnp.evaluate((g.dedw, g.dedb, g.l), x=np.zeros((10)), y=np.zeros(3))






