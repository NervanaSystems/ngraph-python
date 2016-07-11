import geon.backends.graph.funs as be
import geon.backends.graph.analysis as analysis

class Compounded(be.Model):
    def __init__(self, act, L, BS, **kargs):
        super(Compounded, self).__init__(**kargs)
        g = self.graph
        g.L = [be.AxisVar(length=N) for N in L]
        g.BS = be.AxisVar(length=BS)

        g.X = be.placeholder(axes=(g.L[0],g.BS))
        g.Y = be.placeholder(axes=(g.L[-1],g.BS))
        g.W = [be.Variable(axes=(L_np1, L_n), name = 'W%d'%i) for i,(L_np1, L_n) in enumerate(zip(g.L[1:], g.L[:-1]))]
        
        activation = be.tanh
        g.A = [act[0](be.dot(g.W[0],g.X))]
        for i in range(1, len(L)-1):
            g.A.append(act[i](be.dot(g.W[i], g.A[i-1])))
        g.Error = be.cross_entropy_multi(g.A[-1], g.Y)
        
        g.dW = [be.deriv(g.Error, w) for w in g.W]

        dataflow = analysis.DataFlowGraph([g.Error] + g.dW)
        kernelflow = analysis.KernelFlowGraph(dataflow)
        interference = analysis.InterferenceGraph(kernelflow.liveness())
        memory = analysis.color(interference)
        dataflow.view()
        print 'The memory footprint is {} MiB'.format(memory*1024**-2)

activations = [be.tanh, be.tanh, be.softmax]
layers = [1024, 1280, 1280, 10]
bsize=128
Compounded(activations, layers, bsize)
