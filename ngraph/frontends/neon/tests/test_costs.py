import numpy as np
import ngraph as ng
from ngraph.util.utils import ExecutorFactory
from ngraph.frontends.neon.cost import (CrossEntropyBinary, CrossEntropyMulti, SumSquared,
                                        MeanSquared)


def compare_tensors(func, outputs, targets, expected_result, tol=0.):
    ex = ExecutorFactory()
    N = ng.make_axis("N")
    N.length = outputs.shape[0]
    y = ng.placeholder(axes=ng.make_axes([N]))
    t = ng.placeholder(axes=ng.make_axes([N]))

    costfunc = ex.executor(func.__call__(y, t), y, t)
    np.testing.assert_allclose(costfunc(outputs, targets), expected_result, rtol=tol)


"""
    Cross Entropy Binary
"""


def test_cross_entropy_binary(transformer_factory):
    outputs = np.array([0.5, 0.9, 0.1, 0.0001])
    targets = np.array([0.5, 0.99, 0.01, 0.2])
    eps = np.exp(-50)
    expected_log = np.log(np.maximum(outputs, eps))
    expected_mlog = np.log(np.maximum(1 - outputs, eps))
    expected_result = np.sum((-targets * expected_log) - (1 - targets) * expected_mlog,
                             keepdims=True)

    compare_tensors(CrossEntropyBinary(),
                    outputs, targets, expected_result, tol=1e-6)


def test_cross_entropy_binary_limits(transformer_factory):
    outputs = np.array([0.5, 1.0, 0.0, 0.0001])
    targets = np.array(([0.5, 0.0, 1.0, 0.2]))
    eps = np.exp(-50)
    expected_log = np.log(np.maximum(outputs, eps))
    expected_mlog = np.log(np.maximum(1 - outputs, eps))
    expected_result = np.sum((-targets * expected_log) - (1 - targets) * expected_mlog,
                             keepdims=True)

    compare_tensors(CrossEntropyBinary(),
                    outputs, targets, expected_result, tol=1e-6)


"""
    Cross Entropy Multi
"""


def test_cross_entropy_multi(transformer_factory):
    outputs = np.array([0.5, 0.9, 0.1, 0.0001])
    targets = np.array([0.5, 0.99, 0.01, 0.2])
    eps = np.exp(-50)
    expected_log = np.log(np.maximum(outputs, eps))
    expected_result = np.sum(-targets * expected_log, axis=0, keepdims=True)

    compare_tensors(CrossEntropyMulti(),
                    outputs, targets, expected_result, tol=1e-6)


def test_cross_entropy_multi_limits(transformer_factory):
    outputs = np.array([0.5, 1.0, 0.0, 0.0001])
    targets = np.array(([0.5, 0.0, 1.0, 0.2]))
    eps = np.exp(-50)
    expected_log = np.log(np.maximum(outputs, eps))
    expected_result = np.sum(-targets * expected_log, axis=0, keepdims=True)

    compare_tensors(CrossEntropyMulti(),
                    outputs, targets, expected_result, tol=1e-6)


"""
    SumSquared
"""


def test_sum_squared(transformer_factory):
    outputs = np.array([0.5, 0.9, 0.1, 0.0001])
    targets = np.array(([0.5, 0.99, 0.01, 0.2]))
    expected_result = np.sum((outputs - targets) ** 2, axis=0) / 2.
    compare_tensors(SumSquared(), outputs, targets, expected_result, tol=1e-6)


def test_sum_squared_limits(transformer_factory):
    outputs = np.array([0.5, 1.0, 0.0, 0.0001])
    targets = np.array(([0.5, 0.0, 1.0, 0.2]))
    expected_result = np.sum((outputs - targets) ** 2, axis=0) / 2.
    compare_tensors(SumSquared(), outputs, targets, expected_result, tol=1e-7)


"""
    MeanSquared
"""


def test_mean_squared(transformer_factory):
    outputs = np.array([0.5, 0.9, 0.1, 0.0001])
    targets = np.array([0.5, 0.99, 0.01, 0.2])
    expected_result = np.mean((outputs - targets) ** 2, axis=0, keepdims=True) / 2.
    compare_tensors(MeanSquared(), outputs, targets, expected_result, tol=1e-6)


def test_mean_squared_limits(transformer_factory):
    outputs = np.array([0.5, 1.0, 0.0, 0.0001])
    targets = np.array(([0.5, 0.0, 1.0, 0.2]))
    expected_result = np.mean((outputs - targets) ** 2, axis=0, keepdims=True) / 2.
    compare_tensors(MeanSquared(), outputs, targets, expected_result, tol=1e-7)
