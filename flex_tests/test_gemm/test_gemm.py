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

MINIMUM_FLEX_VALUE = -2 ** 15
MAXIMUM_FLEX_VALUE = 2 ** 15 - 1
EPSILON = 0.2

pytestmark = pytest.mark.transformer_dependent("module")


@pytest.mark.parametrize("n, c, d, description", (
    (10, 2, 10, "Vertical (m x 1) multiplied by horizontal (1 x m)"),
    (5, 1, 5, "Horizontal (1 x m) multiplied by vertical(m x 1)"),
    (3, 2, 5, "Horizontal (2 x m) multiplied by vertical (m x 2)"),
    (3, 3, 1, "Horizontal (1 x m) multiplied by square (3 x 3)"),
    (3, 3, 3, "Square (3 x 3) multiplied by square (3 x 3)"),
))
def test_gemm_multiply_matrices(transformer_factory, n, c, d, description):
    """
    :param transformer_factory: to use flex calculations
    :param n: number of columns for second matrix
    :param c: number of columns for first matrix
    :param d: number of rows for first matrix
    :param description: description of a particular test case
    :return: PASS if dot product of flex calculations passes assert_allclose comparing with dot product of numpy,
             FAIL if don't
    """
    print(description)
    template_dot_two_placeholders(n, c, d)

@pytest.mark.parametrize("n, c, const_val, flex_exceptions, description", (
    (3, 3, 1, [[15.99951171875, 15.99951171875, 15.99951171875]],  "Dot product of matrix and positive vector"),
    (9, 9, -0.1, [[-63.19921875, -64, -64, -64, -64, -64, -64, -64, -64]],
     "Dot product of matrix and negative vector"),
    (2, 4, 0, [[0, 0]], "Dot product of matrix and vector of zeros")
))
def test_gemm_multiply_matrix_by_vector(transformer_factory, n, c, const_val, flex_exceptions, description):
    """
    :param transformer_factory: to use flex calculations
    :param n: number of columns for first matrix
    :param c: number of rows for first matrix
    :param const_val: vector is filed using this value
    :param flex_exceptions: each element of the list is the another list of expected values which are caused by
           saturation (expected overflow exists)
    :param description: description of a particular test case
    :return: PASS if dot product of flex calculations passes assert_allclose comparing with dot product of numpy
             or expected overflow occurs and the values are exactly as expected, FAIL if don't
    """
    print (description)
    template_dot_one_placeholder(n, c, const_val, flex_exceptions)


@pytest.mark.parametrize("n, c, scalar, flex_exceptions, description", (
    (3, 3, 0.4, [], "Dot product of matrix and positive scalar"),
    (2, 4, -0.3, [], "Dot product of matrix and negative scalar"),
    (3, 5, 0, [], "Dot product of matrix and zero"),
    (3, 2, 10, [np.array([[0,  63.99804688], [ 63.99804688,  63.99804688], [ 63.99804688,  63.99804688] ])], "Do")
))
def test_gemm_multiply_matrix_by_scalar(transformer_factory, n, c, scalar, flex_exceptions, description):
    """
    :param transformer_factory: to use flex calculations
    :param n: number of rows for matrix
    :param c: number of columns for matrix
    :param scalar: value of scalar which is multiplied
    :param flex_exceptions: each element of the list is the another list of expected values which are caused by
           saturation (expected overflow exists)
    :param description: description of a particular test case
    :return: PASS if dot product of flex calculations passed assert_allclose comparing with dot product of numpy
             The calculation equals:
             ng.dot(scalar, matrix)
    """
    print (description)
    template_dot_one_placeholder_and_scalar(n, c, scalar, flex_exceptions)
