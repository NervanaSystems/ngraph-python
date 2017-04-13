# ----------------------------------------------------------------------------
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
import pytest
import ngraph as ng
from ngraph.testing.flexutil import template_two_placeholders

pytestmark = pytest.mark.transformer_dependent("module")

MINIMUM_FLEX_VALUE = -2 ** 15
MAXIMUM_FLEX_VALUE = 2 ** 15 - 1

x = ng.placeholder(())
z = ng.placeholder(())

bug_1103 = pytest.mark.xfail(strict=True, reason="GitHub issue #1103, "
                                                 "DEC initialization not constrained to allowed range")

test_data_double_operand = (
    # template:(operation, [operand_1, operand_2], expected_result, description

    # test_equal
    bug_1103((ng.equal, [(MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE)], [True],
              "Equality function - underflow expected")),
    bug_1103((ng.equal, [(MAXIMUM_FLEX_VALUE + 2, MAXIMUM_FLEX_VALUE)], [True],
              "Equality function - overflow expected")),
    (ng.equal, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [True],
     "Equality function - negative boundary value equal to negative boundary value"),
    (ng.equal, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [True],
     "Equality function - positive boundary value equal to positive boundary value"),
    (ng.equal, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [False],
     "Equality function - positive boundary value equal to negative boundary value"),
    (ng.equal, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [False],
     "Equality function - negative boundary value equal to positive boundary value"),
    (ng.equal, [(1, 1), (MAXIMUM_FLEX_VALUE, 3.9998779296875), (10, 10)], [True, True, True], "Iterations x == y"),

    # test_not_equal
    bug_1103((ng.not_equal, [(MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE)], [False],
              "Inequality function - underflow expected")),
    bug_1103((ng.not_equal, [(MAXIMUM_FLEX_VALUE + 2, MAXIMUM_FLEX_VALUE)], [False],
              "Inequality function - overflow expected")),
    (ng.not_equal, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [False],
     "Inequality function - negative boundary value not equal to negative boundary value"),
    (ng.not_equal, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [False],
     "Inequality function - positive boundary value not equal to positive boundary value"),
    (ng.not_equal, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [True],
     "Inequality function - positive boundary value not equal to negative boundary value"),
    (ng.not_equal, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [True],
     "Inequality function - negative boundary value not equal to positive boundary value"),
    (ng.not_equal, [(0, 1), (MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE), (MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)],
     [True, True, True], "Iterations x != y"),

    # test_less
    bug_1103((ng.less, [(MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE)], [False], "Less function - underflow expected")),
    bug_1103((ng.less, [(MAXIMUM_FLEX_VALUE + 2, MAXIMUM_FLEX_VALUE)], [True], "Less function - overflow expected")),
    (ng.less, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [False],
     "Less function - negative boundary value less than negative boundary value"),
    (ng.less, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [False],
     "Less function - positive boundary value less than positive boundary value"),
    (ng.less, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [False],
     "Less function - positive boundary value less than negative boundary value"),
    (ng.less, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [True],
     "Less function - negative boundary value less than positive boundary value"),
    (ng.less, [(1, 2), (2, 3), (10000, 10)], (True, True, False), "Iterations x < y"),

    # test_less_equal
    bug_1103((ng.less_equal, [(MAXIMUM_FLEX_VALUE + 2, MAXIMUM_FLEX_VALUE)], [True],
              "Less equal function - overflow expected")),
    (ng.less_equal, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [True],
     "Less equal function - negative boundary value less or equal than negative boundary value"),
    (ng.less_equal, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [True],
     "Less equal function - positive boundary value less or equal than positive boundary value"),
    (ng.less_equal, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [False],
     "Less equal function - positive boundary value less or equal than negative boundary value"),
    (ng.less_equal, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [True],
     "Less equal function - negative boundary value less or equal than positive boundary value"),

    # test_greater
    bug_1103((ng.greater, [(MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE)], [True],
              "Greater function - underflow expected")),
    bug_1103((ng.greater, [(MAXIMUM_FLEX_VALUE + 2, MAXIMUM_FLEX_VALUE)], [False],
              "Greater function - overflow expected")),
    (ng.greater, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [False],
     "Greater function - negative boundary value greater than negative boundary value"),
    (ng.greater, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [False],
     "Greater function - positive boundary value greater than positive boundary value"),
    (ng.greater, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [True],
     "Greater function - positive boundary value greater than negative boundary value"),
    (ng.greater, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [False],
     "Greater function - negative boundary value greater than positive boundary value"),

    # test_greater_equal
    bug_1103((ng.greater_equal, [(MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE)], [True],
              "Greater equal function - underflow expected")),
    (ng.greater_equal, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [True],
     "Greater equal function - negative boundary value greater or equal than negative boundary value"),
    (ng.greater_equal, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [True],
     "Greater equal function - positive boundary value greater or equal than positive boundary value"),
    (ng.greater_equal, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [True],
     "Greater equal function - positive boundary value greater or equal than negative boundary value"),
    (ng.greater_equal, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [False],
     "Greater equal function - negative boundary value greater or equal than positive boundary value")
)

@pytest.mark.parametrize("operation, operand_tuple, expected_result, description", test_data_double_operand)
def test_double_operand(transformer_factory, operation, operand_tuple, expected_result, description):
    template_two_placeholders(operand_tuple, operation(x, z), x, z, expected_result, description)
