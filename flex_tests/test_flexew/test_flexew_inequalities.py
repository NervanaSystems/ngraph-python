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
from ngraph.testing import template_two_placeholders

MINIMUM_FLEX_VALUE = -2 ** 15
MAXIMUM_FLEX_VALUE = 2 ** 15 - 1

x = ng.placeholder(())
z = ng.placeholder(())
bug = pytest.mark.xfail(strict=True)

test_data_double_operand = (
    # template:(operation, operand_1, operand_2, expected_result, description

    # test_equal
    bug((ng.equal, [(MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE)], [1.0], "Equality function - underflow expected")),
    bug((ng.equal, [(MAXIMUM_FLEX_VALUE + 2, MAXIMUM_FLEX_VALUE)], [1.0], "Equality function - overflow expected")),
    (ng.equal, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [1.0],
     "Equality function - negative boundary value equal to negative boundary value"),
    (ng.equal, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [1.0],
     "Equality function - positive boundary value equal to positive boundary value"),
    (ng.equal, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [0.0],
     "Equality function - positive boundary value equal to negative boundary value"),
    (ng.equal, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [0.0],
     "Equality function - negative boundary value equal to positive boundary value"),

    # test_not_equal
    bug((ng.not_equal, [(MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE)], [0.0], "Inequality function - underflow expected")),
    bug((ng.not_equal, [(MAXIMUM_FLEX_VALUE + 2, MAXIMUM_FLEX_VALUE)], [0.0], "Inequality function - overflow expected")),
    (ng.not_equal, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [0.0],
     "Inequality function - negative boundary value not equal to negative boundary value"),
    (ng.not_equal, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [0.0],
     "Inequality function - positive boundary value not equal to positive boundary value"),
    (ng.not_equal, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [1.0],
     "Inequality function - positive boundary value not equal to negative boundary value"),
    (ng.not_equal, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [1.0],
     "Inequality function - negative boundary value not equal to positive boundary value"),

    # test_less
    bug((ng.less, [(MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE)], [0.0], "Less function - underflow expected")),
    bug((ng.less, [(MAXIMUM_FLEX_VALUE + 2, MAXIMUM_FLEX_VALUE)], [1.0], "Less function - overflow expected")),
    (ng.less, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [0.0],
     "Less function - negative boundary value less than negative boundary value"),
    (ng.less, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [0.0],
     "Less function - positive boundary value less than positive boundary value"),
    (ng.less, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [0.0],
     "Less function - positive boundary value less than negative boundary value"),
    (ng.less, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [1.0],
     "Less function - negative boundary value less than positive boundary value"),

    # test_less_equal
    bug((ng.less_equal, [(MAXIMUM_FLEX_VALUE + 2, MAXIMUM_FLEX_VALUE)], [1.0], "Less equal function - overflow expected")),
    (ng.less_equal, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [1.0],
     "Less equal function - negative boundary value less or equal than negative boundary value"),
    (ng.less_equal, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [1.0],
     "Less equal function - positive boundary value less or equal than positive boundary value"),
    (ng.less_equal, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [0.0],
     "Less equal function - positive boundary value less or equal than negative boundary value"),
    (ng.less_equal, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [1.0],
     "Less equal function - negative boundary value less or equal than positive boundary value"),

    # test_greater
    bug((ng.greater, [(MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE)], [1.0], "Greater function - underflow expected")),
    bug((ng.greater, [(MAXIMUM_FLEX_VALUE + 2, MAXIMUM_FLEX_VALUE)], [0.0], "Greater function - overflow expected")),
    (ng.greater, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [0.0],
     "Greater function - negative boundary value greater than negative boundary value"),
    (ng.greater, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [0.0],
     "Greater function - positive boundary value greater than positive boundary value"),
    (ng.greater, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [1.0],
     "Greater function - positive boundary value greater than negative boundary value"),
    (ng.greater, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [0.0],
     "Greater function - negative boundary value greater than positive boundary value"),

    # test_greater_equal
    bug((ng.greater_equal, [(MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE)], [1.0],
         "Greater equal function - underflow expected")),
    (ng.greater_equal, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [1.0],
     "Greater equal function - negative boundary value greater or equal than negative boundary value"),
    (ng.greater_equal, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [1.0],
     "Greater equal function - positive boundary value greater or equal than positive boundary value"),
    (ng.greater_equal, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], [1.0],
     "Greater equal function - positive boundary value greater or equal than negative boundary value"),
    (ng.greater_equal, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], [0.0],
     "Greater equal function - negative boundary value greater or equal than positive boundary value")
)


@pytest.mark.parametrize("operation, operand_tuple, expected_result, description", test_data_double_operand)
def test_double_operand(transformer_factory, operation, operand_tuple, expected_result, description):
    template_two_placeholders(operand_tuple, operation(x, z), x, z, expected_result, description)