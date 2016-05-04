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
        g.x2 = be.dot(g.x, g.x)

        g.z = 2 * be.deriv(be.exp(abs(-be.log(g.x * g.y))), g.x)

    def run(self):
        with be.bound_environment(graph=self) as environment:
            x = np.arange(10, dtype=np.float32) + 1
            y = x * x

            xa = ArrayWithAxes(x, (self.a.S,))
            ya = ArrayWithAxes(y, (self.a.S,))

            gnp = evaluation.GenNumPy(environment=environment, results=(self.naming.x2, self.naming.z, self.naming.w))
            gnp.set_input(self.naming.x, xa)
            gnp.set_input(self.naming.y, ya)
            gnp.evaluate()

            enp = evaluation.NumPyEvaluator(environment=environment, results=(self.naming.x2, self.naming.z, self.naming.w))
            enp.set_input(self.naming.x, xa)
            enp.set_input(self.naming.y, ya)
            resultnp = enp.evaluate()
            print resultnp

            # epc = evaluation.PyCUDAEvaluator(environment=environment, results=(self.naming.z, self.naming.w))
            # epc.set_input(self.naming.x, xa)
            # epc.set_input(self.naming.y, ya)
            # resultpc = epc.evaluate()
            # with cudagpu.cuda_device_context():
            #     print resultpc


e = Eval()
e.run()
