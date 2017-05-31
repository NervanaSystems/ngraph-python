from __future__ import print_function
import numpy as np
import ngraph_shaped as ns
import ngraph as ng

# simple computation
a = ns.placeholder((2, 2))
b = ns.placeholder()
c = ns.constant(3)
d = ns.ones((1, 2)) * 4
f = ns.add(ns.add(ns.add(a, b), c), d)

trans = ng.transformers.make_transformer()
comp = trans.computation(f, a, b)
print(comp(np.ones((2, 2)), 2))

# matmul
a_val = np.random.rand(2, 3)
b_val = np.random.rand(3, 4)
a = ns.constant(a_val)
b = ns.constant(b_val)
f = ns.matmul(a, b)

trans = ng.transformers.make_transformer()
comp = trans.computation(f)
print(comp())
print(np.dot(a_val, b_val))
np.testing.assert_allclose(comp(), np.dot(a_val, b_val))

# reductions
a = ns.ones((2, 3, 4))
f = ns.reduce_sum(a, axis=[0, 2])
trans = ng.transformers.make_transformer()
comp = trans.computation(f)
print(comp())
