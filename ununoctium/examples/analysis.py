import geon.backends.graph.funs as be
import geon.backends.graph.analysis as analysis

class Compounded(be.Model):
    def __init__(self, **kargs):
        super(Compounded, self).__init__(**kargs)
        g = self.graph
        g.N = be.AxisVar()
        g.N.length = 10

        g.X = be.placeholder(axes=(g.N,g.N))
        g.Y = be.placeholder(axes=(g.N,g.N))
        g.W1 = be.placeholder(axes=(g.N,g.N))
        g.W2 = be.placeholder(axes=(g.N,g.N))

        g.A1 = be.tanh(be.dot(g.W1, g.X))
        g.A2 = be.softmax(be.dot(g.W2, g.A1))
        g.Error = be.cross_entropy_multi(g.A2, g.Y)
        
        g.dW1 = be.deriv(g.Error, g.W1)
        g.dW2 = be.deriv(g.Error, g.W2)

        dataflow = analysis.DataFlowGraph([g.dW1, g.dW2])
        kernelflow = analysis.KernelFlowGraph(dataflow)
        interference = analysis.InterferenceGraph(kernelflow.liveness())
        memory = analysis.color(interference)
        print 'The memory footprint is {} GB'.format(memory*10**-9)
        dataflow.view()

Compounded()
