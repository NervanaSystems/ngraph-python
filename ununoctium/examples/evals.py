import geon.backends.graph.graph as graph
import geon.backends.graph.evaluation as evaluation
import numpy as np
import geon.backends.graph.cudagpu as cudagpu
from geon.backends.graph.evaluation import ArrayWithAxes


gr = graph.Graph()
with graph.default_graph(gr) as g:
    import geon.backends.graph.funs as be
    a = be.AxisGenerator('a')
    a.S[10]

    g.x = be.input((a.S,))
    g.y = be.input((a.S,))
    g.w = be.deriv(g.x+g.y, g.y)

    g.z = 2*be.deriv(be.exp(abs(-be.log(g.x * g.y))), g.x)

#graph.show_graph(gr)

x = np.arange(10)+1
y = x*x

xa = ArrayWithAxes(x, (a.S,))
ya = ArrayWithAxes(y, (a.S,))

gnp = evaluation.GenNumPy(graph=gr)
gnp.evaluate((g.z, g.w), x=xa, y=ya)

enp = evaluation.NumPyEnvironment(graph=gr)
resultnp = enp.evaluate((g.z, g.w), x=xa, y=ya)
print resultnp

epc = evaluation.PyCUDAEnvironment(graph=gr)
resultpc = epc.evaluate((g.z, g.w), x=xa, y=ya)
with cudagpu.cuda_device_context():
    print resultpc


