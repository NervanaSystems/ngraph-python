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

import pytest
import numpy as np
from ngraph.testing.flexutil import template_dot_two_placeholders, template_dot_one_placeholder,\
    template_dot_one_placeholder_and_scalar

pytestmark = pytest.mark.flex_only


@pytest.mark.parametrize("rows_1, col_1, col_2, description", (
    (10, 1, 10, "Vertical (m x 1) multiplied by horizontal (1 x m)"),
    (1, 5, 1, "Horizontal (1 x m) multiplied by vertical(m x 1)"),
    (2, 15, 2, "First matrix (2 x m) multiplied by second matrix (m x 2)"),
    (3, 3, 3, "Square (3 x 3) multiplied by square (3 x 3)"),
))
def test_gemm_multiply_matrices(transformer_factory, rows_1, col_1, col_2, description):
    """
    :param [FIXTURE] transformer_factory: py.test fixture to use flex calculations
    :param rows_1: number of rows for first matrix
    :param col_1: number of columns for first matrix
    :param col_2: number of columns for second matrix
    :param description: description of a particular test case
    :return: PASS if dot product of flex calculations passes assert_allclose comparing with dot
    product of numpy, FAIL if don't. Overflow/underflow doesn't occur for above test cases
    Those test cases check only autoflex initialisation - check result of dot product for matrices
    with specific shapes.
    """
    print("Description of test case: ", description)
    template_dot_two_placeholders(col_2, col_1, rows_1)


@pytest.mark.parametrize("row, col, const_val, flex_exceptions, iters, description", (
    (3, 3, 1, [[15.99951171875, 15.99951171875, 15.99951171875]], 3,
     "Dot product of matrix and positive vector"),
    (9, 9, -0.1, [[-63.19921875, -64, -64, -64, -64, -64, -64, -64, -64]], 9,
     "Dot product of matrix and negative vector"), (4, 2, 0, [[0, 0]], 2,
                                                    "Dot product of matrix and vector of zeros")
))
def test_gemm_multiply_matrix_by_vector(transformer_factory, row, col, const_val, flex_exceptions,
                                        iters, description):
    """
    :param [FIXTURE] transformer_factory: py.test fixture to use flex calculations
    :param row: number of rows for first matrix
    :param col: number of columns for first matrix
    :param const_val: vector is filled using this value
    :param flex_exceptions: each element of the list is the another list of expected values which
    are caused by saturation (expected overflow exists)
    :param iters: number of iterations of the same placeholder
    :param description: description of a particular test case
    :return: PASS if dot product of flex calculations passes assert_allclose comparing with dot
    product of numpy or expected overflow occurs and the values are exactly as expected,
    FAIL if don't
    The calculation equals: ng.dot(first_matrix, vector)
    Those test cases check autoflex initialisation and scale adjusting as well.
    """
    print(description)
    template_dot_one_placeholder(row, col, const_val, flex_exceptions, iters)


@pytest.mark.parametrize("row, col, scalar, flex_exceptions, iters, description", (
    (3, 3, 0.4, [], 3, "Dot product of matrix and positive scalar"),
    (2, 4, -0.3, [], 2, "Dot product of matrix and negative scalar"),
    (3, 5, 0, [], 3, "Dot product of matrix and zero"),
    (3, 2, 10, [np.array([[0, 63.99804688], [63.99804688, 63.99804688], [63.99804688,
                                                                         63.99804688]]),
                np.array([[0, 255.9921875], [255.9921875, 255.9921875],
                          [255.9921875, 255.9921875]])], 3,
     "Dot product with two expected overflows")
))
def test_gemm_multiply_matrix_by_scalar(transformer_factory, row, col, scalar, flex_exceptions,
                                        iters, description):
    """
    :param [FIXTURE] transformer_factory: py.test fixture to use flex calculations
    :param row: number of rows for matrix
    :param col: number of columns for matrix
    :param scalar: value of scalar which is multiplied
    :param flex_exceptions: each element of the list is the another list of expected values which
    are caused by saturation (expected overflow exists)
    :param iter: number of iterations of the same placeholder
    :param description: description of a particular test case
    :return: PASS if dot product of flex calculations passed assert_allclose comparing with dot
    product of numpy
    The calculation equals: ng.dot(scalar, matrix)
    Those test cases check autoflex initialisation and scale adjusting as well.
    """
    print(description)
    template_dot_one_placeholder_and_scalar(row, col, scalar, flex_exceptions, iters)
