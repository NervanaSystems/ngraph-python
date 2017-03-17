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
import numpy as np
import pytest
import operator as op

import ngraph as ng
from ngraph.testing import template_one_placeholder

MINIMUM_FLEX_VALUE = -2 ** 15
MAXIMUM_FLEX_VALUE = 2 ** 15 - 1

bug = pytest.mark.xfail(strict=True)
EPSILON = 0.2
x = ng.placeholder(())

test_data_single_operand = (
    # template:(operation, operand, expected_result, description)

    # test_assign
    bug((op.pos, [MINIMUM_FLEX_VALUE - 2], [MINIMUM_FLEX_VALUE], "Assign function - underflow expected")),
    bug((op.pos, [MAXIMUM_FLEX_VALUE + 1], [MAXIMUM_FLEX_VALUE], "Assign function - overflow expected")),
    (op.pos, [MINIMUM_FLEX_VALUE], [MINIMUM_FLEX_VALUE], "Assign function of negative boundary value"),
    (op.pos, [MAXIMUM_FLEX_VALUE], [MAXIMUM_FLEX_VALUE], "Assign function of positive boundary value"),
    (op.pos, [0.4], [0.399993896484375], "Assign function of positive values from flex range - check high precision"),

    # test_neg
    bug((op.neg, [MINIMUM_FLEX_VALUE], [MAXIMUM_FLEX_VALUE], "Negate function - overflow expected")),
    bug((op.neg, [MINIMUM_FLEX_VALUE + 1], [MAXIMUM_FLEX_VALUE],
         "Negate function of negative boundary value inside of flex range")),
    bug((op.neg, [MAXIMUM_FLEX_VALUE], [MINIMUM_FLEX_VALUE + 1],
         "Negate function of positive boundary value inside of flex range")),
    # test_sqrt
    (ng.sqrt, [0], [0], "Square root of zero and zero"),
    (ng.sqrt, [MAXIMUM_FLEX_VALUE], [181.015625], "Square root of positive boundary value"),
    bug((ng.sqrt, [MINIMUM_FLEX_VALUE], [np.nan],
         "Square root of negative boundary value - NaN expected")),

    # test_abs
    bug((ng.absolute, [MINIMUM_FLEX_VALUE], [MAXIMUM_FLEX_VALUE],
         "Absolute value from the flex range - overflow expected")),
    bug((ng.absolute, [MAXIMUM_FLEX_VALUE], [MAXIMUM_FLEX_VALUE], "Absolute value outside of the flex range")),
)

test_data_double_operand = (
    # template:(operation, operand_1, operand_2, expected_result, description)

    # test_add
    bug((op.add, [MAXIMUM_FLEX_VALUE], 1, [MAXIMUM_FLEX_VALUE],
         "Positive boundary value plus one - overflow expected")),
    bug((op.add, [MINIMUM_FLEX_VALUE], 1, [MINIMUM_FLEX_VALUE + 1], "Negative boundary value plus one")),
    (op.add,  [0, 1, 2, 3, 4], 1.5, [1.5, 1.99993896484375, 3.5, 4.5], "x + 1.5"),

    # test_subtraction
    (op.sub, [MINIMUM_FLEX_VALUE], 1, [MINIMUM_FLEX_VALUE], "Negative boundary value minus one - underflow expected"),
    bug((op.sub, [MINIMUM_FLEX_VALUE], 2, [MINIMUM_FLEX_VALUE],
         "Negative boundary value minus two - underflow expected")),
    (op.sub, [MAXIMUM_FLEX_VALUE], 1, [MAXIMUM_FLEX_VALUE - 1], "Positive boundary value minus one"),
    bug((op.sub, [MAXIMUM_FLEX_VALUE], 2, [MAXIMUM_FLEX_VALUE - 2], "Positive boundary value minus two")),

    # test_multiplication
    bug((op.mul, [MINIMUM_FLEX_VALUE], 2, [MINIMUM_FLEX_VALUE],
         "Negative boundary value multiplied by two - underflow expected",)),
    bug((op.mul, [MAXIMUM_FLEX_VALUE], 2, [MAXIMUM_FLEX_VALUE],
         "Positive boundary value multiplied by two - overflow expected",)),
    (op.mul, [MINIMUM_FLEX_VALUE], 0, [0], "Negative boundary value multiplied by zero equals zero"),
    (op.mul, [MAXIMUM_FLEX_VALUE], 1, [MAXIMUM_FLEX_VALUE], "Positive boundary value multiplied by one is the same"),
    (op.mul, [0], 0, [0], "Zero multiplied by zero equals zero"),
    (op.mul, [MINIMUM_FLEX_VALUE], -0.5, [MINIMUM_FLEX_VALUE * (-0.5)],
     "Negative sign value multiplied by negative sign value equals positive sign"),

    # test_division
    bug((op.div, [MAXIMUM_FLEX_VALUE], 0.5, [MAXIMUM_FLEX_VALUE],
         "Positive boundary value division - overflow expected")),
    bug((op.div, [MINIMUM_FLEX_VALUE], 0.5, [MINIMUM_FLEX_VALUE],
         "Negative boundary value division - underflow expected")),
    bug((op.div, [MAXIMUM_FLEX_VALUE], 3, [MAXIMUM_FLEX_VALUE / 3], "Positive boundary value division")),
    bug((op.div, [MINIMUM_FLEX_VALUE], 3, [MINIMUM_FLEX_VALUE / 3], "Negative boundary value division")),

    # test_modulo
    bug((op.mod, [MINIMUM_FLEX_VALUE], 3, [MINIMUM_FLEX_VALUE % 3], "Negative boundary value mod 3")),
    (op.mod, [MAXIMUM_FLEX_VALUE], 3, [MAXIMUM_FLEX_VALUE % 3], "Positive boundary value mod 3"),
    bug((op.mod, [MAXIMUM_FLEX_VALUE], -3, [MAXIMUM_FLEX_VALUE % (-3)], "Positive boundary value mod -3")),

    # test_power
    bug((op.pow, [MAXIMUM_FLEX_VALUE], 2, [MAXIMUM_FLEX_VALUE],
         "Positive boundary value exponentiation - overflow expected")),
    bug((op.pow, [MINIMUM_FLEX_VALUE], 3, [MINIMUM_FLEX_VALUE],
         "Negative boundary value exponentiation - underflow expected")),
    # Not sure of this case, results should be tracked
    (op.pow, [MAXIMUM_FLEX_VALUE], 0.4, [63.99609375], "Positive boundary value exponentiation"),
    (op.pow, [MINIMUM_FLEX_VALUE], -2, [MINIMUM_FLEX_VALUE ** (-2)], "Negative boundary value negative exponentiation")
)


@pytest.mark.parametrize("operation, operand, expected_result, description", test_data_single_operand)
def test_single_operand(transformer_factory, operation, operand, expected_result, description):
    template_one_placeholder(operand, operation(x), x, expected_result, description)


@pytest.mark.parametrize("operation, operand_1, operand_2, expected_result, description", test_data_double_operand)
def test_double_operand(transformer_factory, operation, operand_1, operand_2, expected_result, description):
    template_one_placeholder(operand_1, operation(x, operand_2), x, expected_result, description)
