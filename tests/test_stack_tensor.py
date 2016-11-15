import ngraph as ng
import numpy as np
from ngraph.util.utils import ExecutorFactory, RandomTensorGenerator

delta = 1e-3
rtol = atol = 1e-2


def test_stack():
    ax = ng.make_name_scope(name="ax")
    ax.W = ng.make_axis(length=4)
    ax.H = ng.make_axis(length=5)
    ax.I = ng.make_axis(length=3)

    axes = ng.make_axes([ax.W, ax.H])

    rng = RandomTensorGenerator(0, np.float32)

    a_v = [rng.uniform(0, 1, axes) for i in range(ax.I.length)]

    for pos in range(len(axes) + 1):
        a = [ng.placeholder(axes, initial_value=_) for _ in a_v]

        s = ng.stack(a, ax.I, pos)

        ex = ExecutorFactory()

        num_funs = [ex.numeric_derivative(s, _, delta) for _ in a]
        sym_funs = [ex.derivative(s, _) for _ in a]

        ex.transformer.initialize()

        for n_fun, s_fun, a_i in zip(num_funs, sym_funs, a_v):
            d_n = n_fun(a_i)
            d_s = s_fun(a_i)
            np.allclose(d_n, d_s, rtol=rtol, atol=atol)
