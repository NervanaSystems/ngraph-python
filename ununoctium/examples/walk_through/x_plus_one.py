import geon

x = geon.placeholder(axes=geon.Axes())
x_plus_one = x + 1

transformer = geon.NumPyTransformer()

plus_one = transformer.computation(x_plus_one, x)

for i in range(5):
    print(plus_one(i))
