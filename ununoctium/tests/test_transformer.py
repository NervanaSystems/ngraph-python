import geon as be
import numpy as np


def test_evalutaion_twice():
    """
    test executing a computation graph twice on a one layer MLP
    """
    x = be.NumPyTensor(
        np.array([[1, 2], [3, 4]], dtype='float32'),
        axes=be.Axes([be.NumericAxis(2), be.NumericAxis(2)])
    )

    hidden1_weights = be.NumPyTensor(
        np.array([[1], [1]], dtype='float32'),
        axes=be.Axes([be.NumericAxis(2), be.NumericAxis(1)])
    )

    hidden1_biases = be.NumPyTensor(
        np.array([[2], [2]], dtype='float32'),
        axes=be.Axes([be.NumericAxis(2), be.NumericAxis(1)])
    )

    hidden1 = be.dot(x, hidden1_weights) + hidden1_biases

    trans = be.NumPyTransformer()
    comp = trans.computation([hidden1])
    trans.finalize()

    result_1 = comp.evaluate()[hidden1]
    result_2 = comp.evaluate()[hidden1]
    assert np.array_equal(result_1, result_2)
