import geon.backends.graph.graph as graph
import geon.backends.graph.evaluation as evaluation
import numpy as np
import geon.backends.graph.cudagpu as cudagpu

gr = graph.Graph()
with graph.default_graph(gr) as g:
    import geon.backends.graph.funs as be

    g.x = be.input((10,))
    g.y = be.input((10,))
    g.w = be.deriv(g.x+g.y, g.y)

    g.z = 2*be.deriv(be.exp(abs(-be.log(g.x * g.y))), g.x)

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


