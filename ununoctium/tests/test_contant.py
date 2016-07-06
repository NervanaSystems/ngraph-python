import geon.backends.graph.funs as be
import geon.backends.graph.pycudatransform as evaluation

"""
Test be.Constant

"""
def test_constant_init():
    with be.bound_environment():
        a = be.Constant(5)

        enp = evaluation.NumPyEvaluator(results=[a])
        enp.initialize()

        result = enp.evaluate()[a]
        print(result)

        assert(result == 5)
        print("pass constant initialization")


def test_constant_add():
    with be.bound_environment():
        a = be.Constant(1)
        b = be.Constant(2)
        c = a + b

        enp = evaluation.NumPyEvaluator(results=[c])
        enp.initialize()

        result = enp.evaluate()[c]
        print(result)

        assert(result == 3)
        print("pass constant add")


test_constant_init()
test_constant_add()