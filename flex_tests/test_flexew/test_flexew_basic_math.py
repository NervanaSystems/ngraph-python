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
from __future__ import print_function
import numpy as np
import pytest

import ngraph as ng
from ngraph.testing.flexutil import template_one_placeholder
from ngraph.testing import executor

pytestmark = pytest.mark.transformer_dependent("module")

MINIMUM_FLEX_VALUE = -2 ** 15
MAXIMUM_FLEX_VALUE = 2 ** 15 - 1

# Known issues
bug_1103 = pytest.mark.xfail(strict=True, reason="GitHub issue #1103, "
                                                 "DEC initialization not constrained to allowed range")
bug_1062 = pytest.mark.xfail(strict=True, reason="GitHub issue #1062, problem with ng.sqrt corner cases")
bug_autoflex = pytest.mark.xfail(strict=True, reason="Problem connected to offset in autoflex, to clarify")
bug_abs_max = pytest.mark.xfail(strict=True, reason="Problem connected to absolute max, to clarify")
bug_1064 = pytest.mark.xfail(strict=True, reason="GitHub issue #1064, flex lower priority issues:"
                                                 "modulus and ZeroDivisionError clarification")
bug_1227 = pytest.mark.xfail(strict=True, reason="GitHub issue #1227, find explanation of results")

EPSILON = 0.2
x = ng.placeholder(())


test_assign_data = (
    # template:(operand_to_assign, expected_result, description)

    # test_assign
    bug_1103((MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE, "Assign function - underflow expected")),
    bug_1103((MAXIMUM_FLEX_VALUE + 1, MAXIMUM_FLEX_VALUE, "Assign function - overflow expected")),
    (MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, "Assign function of negative boundary value"),
    bug_1103((MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, "Assign function of positive boundary value")),
    (0.4, 0.399993896484375, "Assign function of positive values from flex range - check high precision"),
)

test_data_single_operand = (
    # template:(operation, operand, expected_result, description)

    # test_neg
    bug_1103((ng.negative, [MINIMUM_FLEX_VALUE], [MAXIMUM_FLEX_VALUE], "Negate function - overflow expected")),
    bug_1103((ng.negative, [MINIMUM_FLEX_VALUE + 1], [MAXIMUM_FLEX_VALUE],
              "Negate function of negative boundary value inside of flex range")),
    bug_1103((ng.negative, [MAXIMUM_FLEX_VALUE], [MINIMUM_FLEX_VALUE + 1],
              "Negate function of positive boundary value inside of flex range")),
    # test_sqrt
    (ng.sqrt, [0], [0], "Square root of zero and zero"),
    (ng.sqrt, [MAXIMUM_FLEX_VALUE], [181.015625], "Square root of positive boundary value"),
    bug_1062((ng.sqrt, [MINIMUM_FLEX_VALUE], [np.nan], "Square root of negative boundary value - NaN expected")),
    (ng.sqrt, [1, 124, 10000, 32000], [1, 1.4141, 2.8283, 5.6567], "Iterations sqrt(x)"),

    # test_abs
    bug_1103((ng.absolute, [MINIMUM_FLEX_VALUE], [MAXIMUM_FLEX_VALUE],
              "Absolute value from the flex range - overflow expected")),
    bug_autoflex((ng.absolute, [MAXIMUM_FLEX_VALUE-2], [MAXIMUM_FLEX_VALUE-2],
                  "Absolute value outside of the flex range")),
    (ng.absolute, [-1, 10000, -0.4, MINIMUM_FLEX_VALUE], [1, 1.9999, 0.3999, 15.9995], "Iterations abs(x)"),


)

test_data_double_operand = (
    # template:(operation, operand_1, operand_2, expected_result, description)

    # test_add
    bug_1103((ng.add, [MAXIMUM_FLEX_VALUE], 2, [MAXIMUM_FLEX_VALUE],
              "Positive boundary value plus one - overflow expected")),
    bug_abs_max((ng.add, [MINIMUM_FLEX_VALUE], 1, [MINIMUM_FLEX_VALUE + 1], "Negative boundary value plus one")),
    (ng.add,  [0, 1, 2, 3, 4], 1.5, [1.5, 1.9999, 3.5, 4.5], "Iterations x + 1.5"),

    # test_subtraction
    (ng.subtract, [MINIMUM_FLEX_VALUE], 1, [MINIMUM_FLEX_VALUE],
     "Negative boundary value minus one - underflow expected"),
    bug_1103((ng.subtract, [MINIMUM_FLEX_VALUE], 2, [MINIMUM_FLEX_VALUE],
              "Negative boundary value minus two - underflow expected")),
    (ng.subtract, [MAXIMUM_FLEX_VALUE], 1, [MAXIMUM_FLEX_VALUE - 1], "Positive boundary value minus one"),
    bug_autoflex((ng.subtract, [MAXIMUM_FLEX_VALUE], 2, [MAXIMUM_FLEX_VALUE - 2], "Positive boundary value minus two")),
    (ng.subtract, [10, 1000, 10000, 1], 0.4, [9.6, 15.5996, 31.999, 0.6015],  "Iterations x - 0.4"),
    (ng.subtract, [10000, 5000, 2500, 1000, 750, 500, 250, 100, 75, 50, 25, 10, 1], 0.4,
     [9999.5, 4999.5, 2500, 1000, 750, 500, 250, 100, 75, 50, 25, 10, 1],  "More complex Iterations x - 0.4"),


    # test_multiplication
    bug_1103((ng.multiply, [MINIMUM_FLEX_VALUE], 2, [MINIMUM_FLEX_VALUE],
              "Negative boundary value multiplied by two - underflow expected",)),
    bug_1103((ng.multiply, [MAXIMUM_FLEX_VALUE], 2, [MAXIMUM_FLEX_VALUE],
              "Positive boundary value multiplied by two - overflow expected",)),
    (ng.multiply, [MINIMUM_FLEX_VALUE], 0, [0], "Negative boundary value multiplied by zero equals zero"),
    (ng.multiply, [MAXIMUM_FLEX_VALUE], 1, [MAXIMUM_FLEX_VALUE],
     "Positive boundary value multiplied by one is the same"),
    (ng.multiply, [0], 0, [0], "Zero multiplied by zero equals zero"),
    (ng.multiply, [MINIMUM_FLEX_VALUE], -0.5, [MINIMUM_FLEX_VALUE * (-0.5)],
     "Negative sign value multiplied by negative sign value equals positive sign"),
    (ng.multiply, [1, 1000, 2, -1000, 0.4], 10.1, [10.0996, 15.9995, 20.1992, -64, 4.0312],
     "Iterations x * 10.1"),

    # test_division
    bug_1103((ng.divide, [MAXIMUM_FLEX_VALUE], 0.5, [MAXIMUM_FLEX_VALUE],
              "Positive boundary value division - overflow expected")),
    bug_1103((ng.divide, [MINIMUM_FLEX_VALUE], 0.5, [MINIMUM_FLEX_VALUE],
              "Negative boundary value division - underflow expected")),
    bug_1227((ng.divide, [MAXIMUM_FLEX_VALUE], 3, [10922], "Positive boundary value division")),
    bug_1227((ng.divide, [MINIMUM_FLEX_VALUE], 3, [-10922], "Negative boundary value division")),
    (ng.divide, [-10, 0.4, MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, 0], 7, [-1.4285, 0.0571, 3.9998, -16, 0],
     "Iterations x / 7"),

    # test_modulo
    bug_1064((ng.mod, [MINIMUM_FLEX_VALUE], 3, [MINIMUM_FLEX_VALUE % 3], "Negative boundary value mod 3")),
    (ng.mod, [MAXIMUM_FLEX_VALUE], 3, [MAXIMUM_FLEX_VALUE % 3], "Positive boundary value mod 3"),
    bug_1064((ng.mod, [MAXIMUM_FLEX_VALUE], -3, [MAXIMUM_FLEX_VALUE % (-3)], "Positive boundary value mod -3")),
    bug_1064((ng.mod, [2.1], 2, [0.09999847412109375], "Modulo of floating point")),

    # test_power
    bug_1103((ng.power, [MAXIMUM_FLEX_VALUE], 2, [MAXIMUM_FLEX_VALUE],
              "Positive boundary value exponentiation - overflow expected")),
    bug_1103((ng.power, [MINIMUM_FLEX_VALUE], 3, [MINIMUM_FLEX_VALUE],
              "Negative boundary value exponentiation - underflow expected")),
    # Not sure of this case, results should be tracked
    (ng.power, [MAXIMUM_FLEX_VALUE], 0.4, [63.99609375], "Positive boundary value exponentiation"),
    (ng.power, [MINIMUM_FLEX_VALUE], -2, [MINIMUM_FLEX_VALUE ** (-2)],
     "Negative boundary value negative exponentiation"),
    (ng.power, [1, 5, -10, 22], 3, [1, 1.9999, -8, 31.999], "Iterations x ^ 3")
)


@pytest.mark.parametrize("operation, operand, expected_result, description", test_data_single_operand)
def test_single_operand(transformer_factory, operation, operand, expected_result, description):
    template_one_placeholder(operand, operation(x), x, expected_result, description)


@pytest.mark.parametrize("operation, operand_1, operand_2, expected_result, description", test_data_double_operand)
def test_double_operand(transformer_factory, operation, operand_1, operand_2, expected_result, description):
    template_one_placeholder(operand_1, operation(x, operand_2), x, expected_result, description)


@pytest.mark.parametrize("operand, expected_result, description", test_assign_data)
def test_assign(transformer_factory, operand, expected_result, description):
    v = ng.variable(())
    vset = ng.sequential([
        ng.assign(v, operand),
        v
    ])
    with executor(vset) as ex:
        print(description)
        vset_ex = ex()
        print("flex: ", vset_ex)
        print("expected: ", expected_result)
        assert vset_ex == expected_result
