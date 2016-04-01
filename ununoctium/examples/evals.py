import geon.backends.graph.graph as graph
import geon.backends.graph.evaluation as evaluation
import numpy as np
import geon.backends.graph.cudagpu as cudagpu

gr = graph.Graph()
with graph.default_graph(gr) as g:
    from geon.backends.graph.funs import *

    g.x = input((10,))
    g.y = input((10,))
    g.w = deriv(g.x+g.y, g.y)

    g.z = 2*deriv(exp(abs(-log(g.x * g.y))), g.x)

#graph.show_graph(gr)

x = np.arange(10)+1
y = x*x

gnp = evaluation.GenNumPy(graph=gr)
gnp.evaluate((g.z, g.w), x=x, y=y)

enp = evaluation.NumPyEnvironment(graph=gr)
resultnp = enp.evaluate((g.z, g.w), x=x, y=y)
print resultnp

epc = evaluation.PyCUDAEnvironment(graph=gr)
resultpc = epc.evaluate((g.z, g.w), x=x, y=y)
with cudagpu.cuda_device_context():
    print resultpc


