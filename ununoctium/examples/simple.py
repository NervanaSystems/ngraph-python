import geon.backends.graph.graph as graph

gr = graph.Graph()
with graph.default_graph(gr) as g:
    from geon.backends.graph.funs import *

    def norm2(x):
        return dot(x.T,x)
    try:
        g.w = zeros((3, 10))
        g.b = zeros((3,))

        g.x = input((10,))
        g.y0 = input((3,))

        g.e = norm2(sig(dot(g.w, g.x) + g.b))
        g.reg = norm2(g.b) + norm2(reshape(g.w, (g.w.size,)))
        g.l = g.e + .1 * g.reg

        g.dedw = deriv(g.l, g.w)
        g.dedb = deriv(g.l, g.b)

    except graph.IncompatibleShapesError:
        # Convenient place to put a breakpoint for debugging
        pass

graph.show_graph(gr)



