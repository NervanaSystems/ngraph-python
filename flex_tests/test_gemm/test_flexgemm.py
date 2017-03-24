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


def test_gemm_1_output_from_flex_range(transformer_factory):
    """
    TODO: make this more interesting
    """
    n, c = 32, 5
    const_val = 0.1

    ng_placeholder, ng_variable = template_create_placeholder_and_variable(n, c, const_val)
    template_one_placeholder(np.ones(c) * const_val, np.ones(n * c).reshape(c, n), ng_placeholder,
                             ng.dot(ng_variable, ng_placeholder))


def test_gemm_multiply_matrix_by_scalar_from_flex_range(transformer_factory):
    """
    Multiply a matrix by a scalar within the flex range
    """
    n, c = 3, 5
    scalar = 0.1

    ar = np.array([i for i in range(c * n)]).reshape(n, c)

    ng_placeholder = template_create_placeholder(n, c)
    ng_var = ng.placeholder(())

    res1 = get_executor_result(scalar, ar, ng_var, ng_placeholder, ng_var * ng_placeholder)
    res2 = np.array([i * scalar for i in ar]).reshape(n, c)

    print("res1\n", res1)
    print("res2\n", res2)

    # assert np.allclose(res1, res2, EPSILON)
    assert_allclose(res1, res2)

# test_data_2 = (
#     (5, 3, 0.1, )
#
# )
# @pytest.mark.parametrize("n, c, const_val, description", test_data_2)

def test_gemm_multiply_matrix_by_negative_matrix_from_flex_range(transformer_factory):
    """
    Multiplies two matrices with values from the flex range (for 8.8 fixed point),
    such as during all calculations there is no over-/underflow.
    """
    n, c = 5, 3
    const_val = 0.1

    ng_placeholder, ng_variable = template_create_placeholder_and_variable(n, c)
    template_one_placeholder(np.ones(c) * const_val, np.array([-i for i in range(c * n)]).reshape(c, n), ng_placeholder,
                             ng.dot(ng_variable, ng_placeholder), lambda a, b: np.dot(a, b))


def test_gemm_multiply_matrix_by_matrix_from_flex_range(transformer_factory):
    """
    Multiplies two matrices with values from the flex range (for 8.8 fixed point),
    such as during all calculations there is no over-/underflow.
    """
    n, c = 5, 5
    const_val = 0.1

    ng_placeholder, ng_variable = template_create_placeholder_and_variable(n, c)
    template_one_placeholder(np.ones(c) * const_val, np.array([i for i in range(c * n)]).reshape(c, n), ng_placeholder,
                             ng.dot(ng_variable, ng_placeholder), lambda a, b: np.dot(a, b))



@pytest.mark.xfail(raises=AssertionError)
def test_gemm_multiply_matrix_by_negative_matrix_below_flex_range(transformer_factory):
    """
    Multiplies two matrices with values from the flex range (for 8.8 fixed point),
    result of the multiplying is below the flex range
    Negative case.
    """
    n, c = 5, 5
    const_val = 0.5

    ng_placeholder, ng_variable = template_create_placeholder_and_variable(n, c, const_val)
    template_one_placeholder(
        np.ones(c) * const_val,
        np.array([i for i in range(int(MINIMUM_FLEX_VALUE) + n, int(MINIMUM_FLEX_VALUE) + n + c * n)]).reshape(c, n),
        ng_placeholder, ng.dot(ng_variable, ng_placeholder), lambda a, b: np.dot(a, b))


def test_gemm_multiply_matrix_by_positive_negative_matrix_from_flex_range(transformer_factory):
    """
    Multiplies two matrices with values from the flex range (for 8.8 fixed point),
    second matrix consists of positive and negative values
    """
    n, c = 5, 5
    const_val = 0.1

    ng_placeholder, ng_variable = template_create_placeholder_and_variable(n, c, const_val)
    template_one_placeholder(np.ones(c) * const_val, np.array([i - (n * c / 2) for i in range(c * n)]).reshape(c, n),
                             ng_placeholder, ng.dot(ng_variable, ng_placeholder), lambda a, b: np.dot(a, b))


@pytest.mark.xfail(raises=AssertionError)
def test_gemm_multiply_matrix_by_matrix_above_flex_range(transformer_factory):
    """
    Multiplies two matrices with values from the flex range (for 8.8 fixed point),
    such as during all calculations there is overflow.
    Negative case.
    """
    n, c = 5, 5
    const_val = 0.5

    ng_placeholder, ng_variable = template_create_placeholder_and_variable(n, c, const_val)
    template_one_placeholder(
        np.ones(c) * const_val,
        np.array([i for i in range(int(MAXIMUM_FLEX_VALUE) - n, int(MAXIMUM_FLEX_VALUE) - n + c * n)]).reshape(c, n),
        ng_placeholder, ng.dot(ng_variable, ng_placeholder), lambda a, b: np.dot(a, b))


@pytest.mark.to_clarify
# @pytest.mark.xfail(raises=AssertionError)
def test_gemm_multiply_matrix_by_outside_flex_matrix_from_flex_range(transformer_factory):
    """
    Multiply two matrices:
    Values of first matrix are within fixed-flex range
    Values of second matrix are outside of the fixed-flex range
    Result od multiplying is within the fixed-flex range
    Negative case ???
    """
    n, c = 5, 5
    const_val = 0.1
    ng_placeholder, ng_variable = template_create_placeholder_and_variable(n, c, const_val)
    template_one_placeholder(
        np.ones(c) * const_val,
        np.array([i for i in range(int(MAXIMUM_FLEX_VALUE), int(MAXIMUM_FLEX_VALUE) + c * n)]).reshape(c, n),
        ng_placeholder, ng.dot(ng_variable, ng_placeholder), lambda a, b: np.dot(a, b))


@pytest.mark.to_clarify
# @pytest.mark.xfail(raises=AssertionError)
def test_multiply_matrix_by_below_flex_matrix_from_flex_range(transformer_factory):
    """
    Multiply two matrices:
    Values of first matrix are within fixed-flex range
    Values of second matrix are below of the fixed-flex range
    Result od multiplying is within the fixed-flex range
    Negative case ???
    """
    n, c = 5, 5
    const_val = 0.1

    ng_placeholder, ng_variable = template_create_placeholder_and_variable(n, c, const_val)
    template_one_placeholder(
        np.ones(c) * const_val,
        np.array([i for i in range(int(MINIMUM_FLEX_VALUE) - c * n, int(MINIMUM_FLEX_VALUE))]).reshape(c, n),
        ng_placeholder, ng.dot(ng_variable, ng_placeholder), lambda a, b: np.dot(a, b))





@pytest.mark.xfail(raises=ValueError)
def test_gemm_multiply_matrix_by_matrix_size_mismatch(transformer_factory):
    """
    Should fail due to mismatch of matrix sizes.
    Negative case.
    """
    n, c, d = 3, 2, 5

    ng_placeholder2, ng_placeholder1 = template_create_placeholders_for_multiplication(n, c, d)
    template_two_placeholders(
        np.array([i for i in range(c * d)]).reshape(c, d),
        np.array([i for i in range(c * n)]).reshape(n, c),
        ng_placeholder1, ng_placeholder2, ng.dot(ng_placeholder1, ng_placeholder2), lambda a, b: 1)


@pytest.mark.problem
# Properties
def test_gemm_from_flex_range_left_distributivity_over_matrix_addition(transformer_factory):
    """
    Check the law of left distributivity over matrix addition.
    Positive case.
    """
    n, c, d = 3, 4, 2

    ng_placeholder2, ng_placeholder1 = template_create_placeholders_for_multiplication(d, c, n)
    ng_placeholder21, ng_placeholder3 = template_create_placeholders_for_addition(c, d)
    ng_placeholder4, ng_placeholder5 = template_create_placeholders_for_addition(n, d)

    arg_array1 = np.array([i / 10.0 for i in range(c * n)]).reshape(n, c)
    arg_array2 = np.array([i / 10.0 for i in range(c * d)]).reshape(c, d)
    arg_array3 = np.array([i / 10.0 for i in range(c * d)]).reshape(c, d)

    sum_cd = get_executor_result(arg_array2, arg_array3, ng_placeholder21, ng_placeholder3,
                                 ng_placeholder21 + ng_placeholder3)

    mul_ncd = get_executor_result(arg_array1, sum_cd, ng_placeholder1, ng_placeholder2,
                                  ng.dot(ng_placeholder1, ng_placeholder2))

    mul_ncd2 = get_executor_result(arg_array1, arg_array2, ng_placeholder1, ng_placeholder2,
                                   ng.dot(ng_placeholder1, ng_placeholder2))
    mul_ncd3 = get_executor_result(arg_array1, arg_array3, ng_placeholder1, ng_placeholder2,
                                   ng.dot(ng_placeholder1, ng_placeholder2))

    sum_nd = get_executor_result(mul_ncd2, mul_ncd3, ng_placeholder4, ng_placeholder5,
                                 ng_placeholder4 + ng_placeholder5)

    assert np.allclose(mul_ncd, sum_nd, EPSILON)


def test_gemm_from_flex_range_right_distributivity_over_matrix_addition(transformer_factory):
    """
    Check the law of right distributivity over matrix addition.
    Positive case.
    """
    n, c, d = 3, 4, 2

    ng_placeholder2, ng_placeholder1 = template_create_placeholders_for_multiplication(d, c, n)
    ng_placeholder21, ng_placeholder3 = template_create_placeholders_for_addition(n, c)
    ng_placeholder4, ng_placeholder5 = template_create_placeholders_for_addition(n, d)

    arg_array1 = np.array([i / 10.0 for i in range(c * n)]).reshape(n, c)
    arg_array2 = np.array([i / 10.0 for i in range(c * n)]).reshape(n, c)
    arg_array3 = np.array([i / 10.0 for i in range(c * d)]).reshape(c, d)

    sum_nc = get_executor_result(arg_array1, arg_array2, ng_placeholder21, ng_placeholder3,
                                 ng_placeholder21 + ng_placeholder3)

    mul_ncd = get_executor_result(sum_nc, arg_array3, ng_placeholder1, ng_placeholder2,
                                  ng.dot(ng_placeholder1, ng_placeholder2))

    mul_ncd2 = get_executor_result(arg_array1, arg_array3, ng_placeholder1, ng_placeholder2,
                                   ng.dot(ng_placeholder1, ng_placeholder2))
    mul_ncd3 = get_executor_result(arg_array2, arg_array3, ng_placeholder1, ng_placeholder2,
                                   ng.dot(ng_placeholder1, ng_placeholder2))

    sum_nd = get_executor_result(mul_ncd2, mul_ncd3, ng_placeholder4, ng_placeholder5,
                                 ng_placeholder4 + ng_placeholder5)

    assert np.allclose(mul_ncd, sum_nd, EPSILON)


def test_gemm_from_flex_range_scalar_mul_distributivity(transformer_factory):
    """
    Checks distributivity of multiplication of scalar and a matrix.
    Positive case.
    """
    n, c, d = 2, 3, 4

    scalar = 0.1
    ar1 = np.array([i for i in range(c * d)]).reshape(d, c)
    ar2 = np.array([i for i in range(c * n)]).reshape(c, n)

    ng_placeholder2, ng_placeholder1 = template_create_placeholders_for_multiplication(n, c, d)
    ng_placeholder_dc = template_create_placeholder(d, c)
    ng_placeholder_dn = template_create_placeholder(d, n)
    ng_var = ng.placeholder(())

    mul1 = get_executor_result(scalar, ar1, ng_var, ng_placeholder_dc, ng_var * ng_placeholder_dc)
    res1 = get_executor_result(mul1, ar2, ng_placeholder1, ng_placeholder2, ng.dot(ng_placeholder1, ng_placeholder2))

    dot2 = get_executor_result(ar1, ar2, ng_placeholder1, ng_placeholder2, ng.dot(ng_placeholder1, ng_placeholder2))
    res2 = get_executor_result(scalar, dot2, ng_var, ng_placeholder_dn, ng_var * ng_placeholder_dn)

    assert np.allclose(res1, res2, EPSILON)


def test_gemm_from_flex_range_transpose_mul_distributivity(transformer_factory):
    """
    Checks the distributivity law of multiplication of two matrices and their transposes.
    Positive case.
    """
    n, c, d = 2, 3, 4

    ar1 = np.array([i - 0.3 for i in range(c * d)]).reshape(d, c)
    ar2 = np.array([i + 0.1 for i in range(c * n)]).reshape(c, n)

    ng_placeholder2, ng_placeholder1 = template_create_placeholders_for_multiplication(n, c, d)
    ng_placeholder4, ng_placeholder3 = template_create_placeholders_for_multiplication(d, c, n)

    dot1 = get_executor_result(ar1, ar2, ng_placeholder1, ng_placeholder2, ng.dot(ng_placeholder1, ng_placeholder2))
    res1 = np.transpose(dot1)

    ar1 = np.transpose(ar1)
    ar2 = np.transpose(ar2)
    res2 = get_executor_result(ar2, ar1, ng_placeholder3, ng_placeholder4, ng.dot(ng_placeholder3, ng_placeholder4))

    assert np.allclose(res1, res2, EPSILON)


def test_gemm_from_flex_range_identity_mul_distributivity(transformer_factory):
    """
    Checks the distributivity of multiplication of an array with an identity array.
    Positive case.
    """
    n, c, d = 3, 3, 3

    ar2 = np.eye(n)
    ar1 = np.array([i + 0.3 for i in range(c * d)]).reshape(d, c)
    ar2 = np.array(ar2).reshape(c, n)

    ng_placeholder2, ng_placeholder1 = template_create_placeholders_for_multiplication(n, c, d)

    res1 = get_executor_result(ar1, ar2, ng_placeholder1, ng_placeholder2, ng.dot(ng_placeholder1, ng_placeholder2))
    res2 = get_executor_result(ar2, ar1, ng_placeholder1, ng_placeholder2, ng.dot(ng_placeholder1, ng_placeholder2))

    assert np.allclose(res1, ar1, EPSILON)
    assert np.allclose(res2, ar1, EPSILON)


def test_gemm_from_flex_range_inverse_mul(transformer_factory):
    """
    Multiplies two square matrices, one of which is the inverse of the other and checks
    if the result is an identity matrix.
    Positive case.
    """
    n = c = d = 3

    id_m = np.eye(n)
    ar = np.random.uniform(-5, 5, c * c).reshape(d, c)
    inv_ar = np.linalg.inv(ar)
    id_m = np.array(id_m).reshape(c, n)

    ng_placeholder2, ng_placeholder1 = template_create_placeholders_for_multiplication(n, c, d)

    res = get_executor_result(ar, inv_ar, ng_placeholder1, ng_placeholder2, ng.dot(ng_placeholder1, ng_placeholder2))

    assert np.allclose(res, id_m, EPSILON)


def test_gemm_from_flex_range_determinant_mul(transformer_factory):
    """
    Checks if determinant of dot product of two square matrices is the same as product of their determinants.
    Positive case.
    """
    n, c, d = 3, 3, 3

    ar1 = np.array([i for i in range(c * d)]).reshape(d, c)
    ar2 = np.array([i - 0.3 for i in range(c * d)]).reshape(c, n)

    ng_placeholder2, ng_placeholder1 = template_create_placeholders_for_multiplication(n, c, d)

    res1 = np.linalg.det(get_executor_result(ar1, ar2, ng_placeholder1, ng_placeholder2, ng.dot(ng_placeholder1, ng_placeholder2)))
    res2 = np.linalg.det(ar1) * np.linalg.det(ar2)

    assert np.allclose(res1, res2, EPSILON)