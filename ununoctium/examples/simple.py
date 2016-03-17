import geon.backends.graph.graph as g

gr = g.Graph()
with g.default_graph(gr) as be:
    from geon.backends.graph.funs import *
    vars = g.VariableBlock()

    def norm2(x):
        return dot(x.T,x)
    try:
        vars.w = zeros((3,10))
        vars.b = zeros((3,))

        vars.x = input((10,))
        vars.y0 = input((3,))

        vars.e = norm2(sin(dot(vars.w, vars.x) + vars.b))
        vars.reg = norm2(vars.b)+norm2(reshape(vars.w,(30,)))
        vars.l = vars.e+.1*vars.reg

        vars.dedw = deriv(vars.l, vars.w)
        vars.dedb = deriv(vars.l, vars.b)

    except g.IncompatibleShapesError:
        # Convenient place to put a breakpoint for debugging
        pass

g.show_graph(gr)



