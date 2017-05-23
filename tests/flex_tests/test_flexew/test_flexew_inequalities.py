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
from ngraph.testing.flexutil import template_two_placeholders, MINIMUM_FLEX_VALUE, \
    MAXIMUM_FLEX_VALUE, id_func

pytestmark = pytest.mark.transformer_dependent("module")

test_data_double_operand = (
    # template:(ng_operation, [(operand_1, operand_2, expected_result, *case_description)],
    # test_description),
    # *case_description is optional

    # test_equal
    (ng.equal, [(MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE, True)],
     "Equality function - underflow expected"),
    (ng.equal, [(MAXIMUM_FLEX_VALUE + 2, MAXIMUM_FLEX_VALUE, True)],
     "Equality function - overflow expected"),
    (ng.equal, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, True)],
     "Equality function - negative boundary value equal to negative boundary value"),
    (ng.equal, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, True)],
     "Equality function - positive boundary value equal to positive boundary value"),
    (ng.equal, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, False)],
     "Equality function - positive boundary value equal to negative boundary value"),
    (ng.equal, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, False)],
     "Equality function - negative boundary value equal to positive boundary value"),
    (ng.equal, [(1, 1, True),
                (MAXIMUM_FLEX_VALUE, 4, True, "Operand 1 and operand 2 overflow to 1.9999"),
                (10, 12, True, "Operand 1 and operand 2 overflow to 7.9997")],
     "Iterations x == y"),

    # test_not_equal
    (ng.not_equal, [(MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE, False)],
     "Inequality function - underflow expected"),
    (ng.not_equal, [(MAXIMUM_FLEX_VALUE + 2, MAXIMUM_FLEX_VALUE, False)],
     "Inequality function - overflow expected"),
    (ng.not_equal, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, False)],
     "Inequality function - negative boundary value not equal to negative boundary value"),
    (ng.not_equal, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, False)],
     "Inequality function - positive boundary value not equal to positive boundary value"),
    (ng.not_equal, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, True)],
     "Inequality function - positive boundary value not equal to negative boundary value"),
    (ng.not_equal, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, True)],
     "Inequality function - negative boundary value not equal to positive boundary value"),
    (ng.not_equal, [(0, 1, True),
                    (MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, True, "Operand 1 underflow to -128, "
                                                                   "operand 2 underflow to -2"),
                    (MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, True, "Operand 1 underflow to -512"
                                                                   " operand 2 underflow to -8")],
     "Iterations x != y"),

    # test_less
    (ng.less, [(MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE, False)],
     "Less function - underflow expected"),
    (ng.less, [(MAXIMUM_FLEX_VALUE + 2, MAXIMUM_FLEX_VALUE, False)],
     "Less function - overflow expected"),
    (ng.less, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, False)],
     "Less function - negative boundary value less than negative boundary value"),
    (ng.less, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, False)],
     "Less function - positive boundary value less than positive boundary value"),
    (ng.less, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, False)],
     "Less function - positive boundary value less than negative boundary value"),
    (ng.less, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, True)],
     "Less function - negative boundary value less than positive boundary value"),
    (ng.less, [(0.0049, 0.005, True),
               (2, 3, False, "Operand 1 and operand 2 overflow to 0.00781226158142"),
               (10, MAXIMUM_FLEX_VALUE, False,
                "Operand 1 and operand 2 overflow to 0.0312490463257")], "Iterations x < y"),

    # test_less_equal
    (ng.less_equal, [(MAXIMUM_FLEX_VALUE + 2, MAXIMUM_FLEX_VALUE, True)],
     "Less equal function - overflow expected"),
    (ng.less_equal, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, True)],
     "Less equal function - negative boundary value less or equal than negative boundary value"),
    (ng.less_equal, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, True)],
     "Less equal function - positive boundary value less or equal than positive boundary value"),
    (ng.less_equal, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, False)],
     "Less equal function - positive boundary value less or equal than negative boundary value"),
    (ng.less_equal, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, True)],
     "Less equal function - negative boundary value less or equal than positive boundary value"),
    (ng.less_equal, [(10000, 10000, True),
                     (9999, 10000, True),
                     (0.4, 0.1, True, "Operand 1 and operand 2 overflow to 0")],
     "Iterations x <= y"),

    # test_greater
    (ng.greater, [(MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE, False)],
     "Greater function - underflow expected"),
    (ng.greater, [(MAXIMUM_FLEX_VALUE + 2, MAXIMUM_FLEX_VALUE, False)],
     "Greater function - overflow expected"),
    (ng.greater, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, False)],
     "Greater function - negative boundary value greater than negative boundary value"),
    (ng.greater, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, False)],
     "Greater function - positive boundary value greater than positive boundary value"),
    (ng.greater, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, True)],
     "Greater function - positive boundary value greater than negative boundary value"),
    (ng.greater, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, False)],
     "Greater function - negative boundary value greater than positive boundary value"),
    (ng.greater, [(1, 0.4, True),
                  (10000, 10000, True,
                   "Operand 1 overflow to 1.99993896484 and operand 2 overflow to 0.49998"),
                  (10000, 10000, True,
                   "Operand 1 overflow to 7.9997 and operand 2 overflow to 1.9999")],
     "Iterations x > y"),

    # test_greater_equal
    (ng.greater_equal, [(MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE, True)],
     "Greater equal function - underflow expected"),
    (ng.greater_equal, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, True)],
     "Greater equal function - negative boundary value greater or equal "
     "than negative boundary value"),
    (ng.greater_equal, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, True)],
     "Greater equal function - positive boundary value greater or equal "
     "than positive boundary value"),
    (ng.greater_equal, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, True)],
     "Greater equal function - positive boundary value greater or equal "
     "than negative boundary value"),
    (ng.greater_equal, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, False)],
     "Greater equal function - negative boundary value greater or equal "
     "than positive boundary value"),
    (ng.greater_equal, [(1, 1, True),
                        (9000, 10000, True, "Operand 1 and operand 2 overflow to 1.9999"),
                        (9001, 10000, True, "Operand 1 and operand 2 overflow to 7.9997")],
     "Iterations x >= y"),
)


@pytest.mark.parametrize("operation, operands, test_name", test_data_double_operand, ids=id_func)
def test_double_operand(transformer_factory, operation, operands, test_name):
    template_two_placeholders(operands, operation)
