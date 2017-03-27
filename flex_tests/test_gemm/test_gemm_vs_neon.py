# -----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
import pytest

import ngraph as ng
from ngraph.testing import executor, assert_allclose
from ngraph.testing.template import template_create_placeholders_for_multiplication, \
    template_dot_two_placeholders, template_create_placeholder_and_variable, template_dot_one_placeholder, \
    template_create_placeholder, get_executor_result

# matrix multiply
MINIMUM_FLEX_VALUE = -2 ** 15
MAXIMUM_FLEX_VALUE = 2 ** 15 - 1
EPSILON = 0.2

pytestmark = pytest.mark.transformer_dependent("module")




matrices_to_multiply = (
    (1, 5, 1, "Vertical (m x 1) multiplied by horizontal (1 x m)"),
    (5, 1, 5, "Horizontal (1 x m) multiplied by vertical(m x 1)"),
    (3, 2, 5, "Horizontal (2 x m) multiplied by vertical (m x 2)"),
    (3, 3, 1, "Horizontal (1 x m) multiplied by square (3 x 3)"),
    (3, 3, 3, "Square (3 x 3) multiplied by square (3 x 3)"),
)


@pytest.mark.parametrize("n, c, d, description", matrices_to_multiply)
def test_gemm_multiply_matrices(transformer_factory, n, c, d, description):
    """
    Multiplies two matrices within the fixed-flex range
    """
    print(description)
    ng_placeholder2, ng_placeholder1 = template_create_placeholders_for_multiplication(n, c, d)
    template_dot_two_placeholders(
        np.array([i for i in range(c * d)]).reshape(d, c),
        np.array([i for i in range(c * n)]).reshape(c, n),
        ng_placeholder1, ng_placeholder2, ng.dot(ng_placeholder1, ng_placeholder2), lambda a, b: np.dot(a, b))


def test_gemm_multiply_matrix_by_variable_matrix(transformer_factory):
    """
    Multiplies two matrices with values from the flex range (for 8.8 fixed point),
    such as during all calculations there is no over-/underflow.
    """
    n, c = 5, 5
    const_val = 0.1

    ng_placeholder, ng_variable = template_create_placeholder_and_variable(n, c)
    template_dot_one_placeholder(np.ones(c) * const_val, np.array([i for i in range(c * n)]).reshape(c, n),
                                 ng_placeholder, ng.dot(ng_variable, ng_placeholder), lambda a, b: np.dot(a, b))


def test_gemm_multiply_matrix_by_scalar(transformer_factory):
    """
    Multiply a matrix by a scalar within the flex range
    """
    n, c = 3, 5
    scalar = 0.1

    ar = np.array([i for i in range(c * n)]).reshape(n, c)

    ng_placeholder = template_create_placeholder(n, c)
    ng_var = ng.placeholder(())

    res1 = get_executor_result(scalar, ar, ng_var, ng_placeholder, ng.dot(ng_var, ng_placeholder))
    res2 = np.array([i * scalar for i in ar]).reshape(n, c)

    print("res1\n", res1)
    print("res2\n", res2)

    assert_allclose(res1, res2)


# matrix multiply
def test_gemm(transformer_factory):
    """
    TODO: make this more interesting
    """
    n, c = 3, 3

    N = ng.make_axis(length=n, name='N')
    C = ng.make_axis(length=c)

    X = ng.placeholder(axes=[C, N])
    Y = ng.placeholder(axes=[N])

    W = ng.variable(axes=[C], initial_value=0.1)

    Y_hat = ng.dot(W, X)
    w = np.ones(c) * 0.1
    xs = np.ones(n * c).reshape(c, n)

    with executor(Y_hat, X) as ex:
        mm_executor = ex



        for ii in range(3):

            y_hat_val = mm_executor(xs)
            print (np.dot(xs, w))
            print (y_hat_val)
            assert_allclose(np.dot(xs, w), y_hat_val)



#https://ngraph.nervanasys.com/docs/building_graphs.html
def test_assign(transformer_factory):
    from ngraph.transformers.nptransform import NumPyTransformer
    w = ng.variable((), initial_value=0)

    a = ng.assign(w, w + 5)
    transformer = NumPyTransformer()
    w_comp = transformer.computation(a)
    print(w_comp())
    print(w_comp())
    # with executor(a) as ex:
    #     res = ex()
    #     print (res)


def test_assign2():
    v = ng.variable(())

    vset2 = ng.sequential([
        ng.assign(v, 99),
        v
    ])
    with executor(vset2) as ex:
        e_v12 = ex()
        print (e_v12)