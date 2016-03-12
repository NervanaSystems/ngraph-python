import geon.backends.graph.graph as graph

def norm2(x):
    return x.T*x

be = graph.Graph()

w = be.input('w')
b = be.input('b')

x = be.input('x')
y0 = be.input('y0')

e = norm2(w * x + b)

dedw = be.deriv(e,w)
dedb = be.deriv(e,b)

graph.show_graph(be)



