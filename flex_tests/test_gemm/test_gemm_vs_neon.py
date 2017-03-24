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

# matrix multiply
MINIMUM_FLEX_VALUE = -2 ** 15
MAXIMUM_FLEX_VALUE = 2 ** 15 - 1
EPSILON = 0.2


def get_executor_result(arg_array1, arg_array2, ng_placeholder1, ng_placeholder2, ng_fun):
    with executor(ng_fun, ng_placeholder1, ng_placeholder2) as m_executor:
        print('\nfun\n', ng_fun)

        print('\narg_array1\n', arg_array1)
        print('\narg_array2\n', arg_array2)

        print('\np1', ng_placeholder1.shape)
        print('\np2', ng_placeholder2.shape)

        result = m_executor(arg_array1, arg_array2)
        print('\nresult\n', result)

        return result


def template_create_placeholder_and_variable(n, c, const_val=0.1):
    # ax = ng.make_name_scope().named('ax')
    N = ng.make_axis(length=n,  name="N")
    C = ng.make_axis(length=c, name="C")

    X = ng.placeholder(axes=(C, N))
    # W = ng.variable(axes=[C - 1], initial_value=const_val)

    W = ng.variable(axes=[C], initial_value=const_val)

    return X, W


def template_create_placeholder(n, c):
    # ax = ng.make_name_scope().named('ax')
    N = ng.make_axis(length=n, name="N")
    C = ng.make_axis(length=c, name="C")

    return ng.placeholder((N, C))


def template_create_placeholders_for_multiplication(n, c, d):
    # ax = ng.make_name_scope().named('ax')
    N = ng.make_axis(length=n, name="N")
    C = ng.make_axis(length=c, name="C")
    D = ng.make_axis(length=d, name="D")

    X = ng.placeholder((C, N))
    # Y = ng.placeholder((D, C - 1))
    Y = ng.placeholder((D, C))
    return X, Y


def template_create_placeholders_for_addition(n, c):
    # ax = ng.make_name_scope().named('ax')
    N = ng.make_axis(length=n, name="N")
    C = ng.make_axis(length=c, name="C")

    X = ng.placeholder((N, C))
    Y = ng.placeholder((N, C))

    return X, Y


def template_one_placeholder(var_array, arg_array, ng_placeholder, ng_fun, fun=lambda a, b: np.dot(a, b),
                             epsilon=EPSILON):
    with executor(ng_fun, ng_placeholder) as mm_executor:
        print('var_array\n', var_array)
        print('arg_array\n', arg_array)

        print('ph\n', ng_placeholder.shape)

        ng_op_out = mm_executor(arg_array)
        np_op_out = fun(var_array, arg_array)

        print('np_op_out\n', np_op_out)
        print(len(np_op_out))
        print('ng_op_out\n', ng_op_out)
        print(len(ng_op_out))
        # 8.8 fixed point test
        # assert assert_allclose(ng_op_out, np_op_out)
        assert_allclose(ng_op_out, np_op_out)


        # assert


def template_two_placeholders(arg_array1, arg_array2, ng_placeholder1, ng_placeholder2, ng_fun,
                              fun=lambda a, b: np.dot(a, b), epsilon=EPSILON):
    with executor(ng_fun, ng_placeholder1, ng_placeholder2) as mm_executor:
        print('arg_array1\n', arg_array1)
        print('arg_array2\n', arg_array2)

        print('ph1\n', ng_placeholder1.shape)
        print('ph2\n', ng_placeholder2.shape)

        np_op_out = fun(arg_array1, arg_array2)
        ng_op_out = mm_executor(arg_array1, arg_array2)


        print('np_op_out\n', np_op_out)
        print('ng_op_out\n', ng_op_out)

        # 8.8 fixed point test
        assert_allclose(ng_op_out, np_op_out)


test_data = (
    (1, 5, 1, "Vertical (m x 1) multiplied by horizontal (1 x m)"),
    (5, 1, 5, "Horizontal (1 x m) multiplied by vertical(m x 1)"),
    (3, 2, 5, "Horizontal (2 x m) multiplied by vertical (m x 2)"),
    (3, 3, 1, "Horizontal (1 x m) multiplied by square (3 x 3)"),
    (3, 3, 3, "Square (3 x 3) multiplied by square (3 x 3)"),
)

@pytest.mark.parametrize("n, c, d, description", test_data)
def test_gemm_multiply_matrices(transformer_factory, n, c, d, description):
    """
    Multiplies two matrices within the fixed-flex range:
    first vertical (m x 1) and second horizontal (1 x m)
    """
    print(description)
    ng_placeholder2, ng_placeholder1 = template_create_placeholders_for_multiplication(n, c, d)
    template_two_placeholders(
        np.array([i for i in range(c * d)]).reshape(d, c),
        np.array([i for i in range(c * n)]).reshape(c, n),
        ng_placeholder1, ng_placeholder2, ng.dot(ng_placeholder1, ng_placeholder2), lambda a, b: np.dot(a, b))


