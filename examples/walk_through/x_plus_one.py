from __future__ import print_function
import ngraph as ng
import ngraph.transformers as ngt

x = ng.placeholder(axes=ng.make_axes())
x_plus_one = x + 1

transformer = ngt.make_transformer()

plus_one = transformer.computation(x_plus_one, x)

for i in range(5):
    print(plus_one(i))
