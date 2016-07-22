from geon.backends.graph.graph_test_utils import execute, be, Axes,\
    in_bound_environment
import numpy as np


@in_bound_environment
def test_dot_with_numerics():
    ax1 = be.NumericAxis(2)
    ax2 = be.NumericAxis(2)
    axes = Axes(ax1, ax2)

    x_np = np.array([[1, 2], [1, 2]], dtype='float32')
    x = be.NumPyTensor(x_np, axes=axes)

    d = be.dot(x, x, numpy_matching=True)
    d_val, = execute([d])

    assert np.array_equal(d_val, np.dot(x_np, x_np))

if __name__ == '__main__':
    test_dot_with_numerics()
