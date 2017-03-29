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
    template_dot_two_placeholders, template_dot_one_placeholder, \
    template_create_placeholder, get_executor_result

# matrix multiply
MINIMUM_FLEX_VALUE = -2 ** 15
MAXIMUM_FLEX_VALUE = 2 ** 15 - 1
EPSILON = 0.2

pytestmark = pytest.mark.transformer_dependent("module")


@pytest.mark.parametrize("n, c, d, description", (
    # template:  (dimension_1, dimension_2, dimension_3, description)

    (1, 5, 1, "Vertical (m x 1) multiplied by horizontal (1 x m)"),
    (5, 1, 5, "Horizontal (1 x m) multiplied by vertical(m x 1)"),
    (3, 2, 5, "Horizontal (2 x m) multiplied by vertical (m x 2)"),
    (3, 3, 1, "Horizontal (1 x m) multiplied by square (3 x 3)"),
    (3, 3, 3, "Square (3 x 3) multiplied by square (3 x 3)"),
))
def test_gemm_multiply_matrices(transformer_factory, n, c, d, description):
    print(description)
    ng_placeholder2, ng_placeholder1 = template_create_placeholders_for_multiplication(n, c, d)
    template_dot_two_placeholders(
        np.array([i for i in range(c * d)]).reshape(d, c),
        np.array([i for i in range(c * n)]).reshape(c, n),
        ng_placeholder1, ng_placeholder2, ng.dot(ng_placeholder1, ng_placeholder2), lambda a, b: np.dot(a, b))


@pytest.mark.parametrize("n, c, const_val, expected_result, description", (
    # template:  (dimension_1, dimension_2, vector_value, flex_exceptions,  description)

    (3, 3, 1, [[15.99951171875, 15.99951171875, 15.99951171875]],  "Dot product of matrix and positive vector"),
    (9, 9, -0.1, [[-63.19921875, -64, -64, -64, -64, -64, -64, -64, -64]],
     "Dot product of matrix and negative vector"),
    (2, 4, 0, [[0, 0]], "Dot product of matrix and vector of zeros")
))
def test_gemm_multiply_matrix_by_vector(transformer_factory, n, c, const_val, expected_result, description):
    print (description)
    template_dot_one_placeholder(n, c, const_val, expected_result)


@pytest.mark.parametrize("n, c, scalar, description", (
    # template:  (dimension_1, dimension_2, scalar, description)

    (3, 3, 0.4, "Dot product of matrix and positive scalar"),
    (2, 4, -0.3, "Dot product of matrix and negative scalar"),
    (3, 5, 0, "Dot product of matrix and zero")
))
def test_gemm_multiply_matrix_by_scalar(transformer_factory, n, c, scalar, description):
    print (description)
    ar = np.array([i for i in range(c * n)]).reshape(n, c)

    ng_placeholder = template_create_placeholder(n, c)
    ng_var = ng.placeholder(())

    res1 = get_executor_result(scalar, ar, ng_var, ng_placeholder, ng.dot(ng_var, ng_placeholder))
    res2 = np.array([i * scalar for i in ar]).reshape(n, c)

    print("res1\n", res1)
    print("res2\n", res2)

    assert_allclose(res1, res2)
