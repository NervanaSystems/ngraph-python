import geon.backends.graph.graph as graph
import geon.backends.graph.evaluation as evaluation
import numpy as np
import geon.backends.graph.cudagpu as cudagpu
from geon.backends.graph.evaluation import ArrayWithAxes
import geon.backends.graph.funs as be

class Eval(be.Model):
    def __init__(self, **kargs):
        super(Eval, self).__init__(**kargs)
        a = self.a
        a.S[10]
        g = self.naming

        g.x = be.input()
        g.y = be.input()
        g.w = be.deriv(g.x + g.y, g.y)

        g.z = 2 * be.deriv(be.exp(abs(-be.log(g.x * g.y))), g.x)

    def run(self):
        with be.bound_environment(graph=self) as environment:
            x = np.arange(10) + 1
            y = x * x

            xa = ArrayWithAxes(x, (self.a.S,))
            ya = ArrayWithAxes(y, (self.a.S,))

            gnp = evaluation.GenNumPy(graph=self, results=(self.naming.z, self.naming.w))
            gnp.evaluate(x=xa, y=ya)

            enp = evaluation.NumPyEnvironment(graph=self, results=(self.naming.z, self.naming.w))
            resultnp = enp.evaluate(x=xa, y=ya)
            print resultnp

            epc = evaluation.PyCUDAEnvironment(graph=self, results=(self.naming.z, self.naming.w))
            resultpc = epc.evaluate(x=xa, y=self)
            with cudagpu.cuda_device_context():
                print resultpc


e = Eval()
e.run()
