import geon.backends.graph.funs as be
import geon.backends.graph.analysis as analysis

class Compounded(be.Model):
    def __init__(self, **kargs):
        super(Compounded, self).__init__(**kargs)
        g = self.graph
        g.S = be.AxisVar()
        g.S.length = 10

        g.x = be.placeholder(axes=(g.S,))
        g.y = be.placeholder(axes=(g.S,))
        g.w = be.deriv(g.x + g.y, g.y)
        g.x2 = be.dot(g.x, g.x)

        g.z = 2 * be.deriv(be.exp(abs(-be.log(g.x * g.y))), g.x)

        dataflow = analysis.DataFlowGraph([g.z])
        kernelflow = analysis.KernelFlowGraph(dataflow)
        kernelflow.view()

Compounded()
