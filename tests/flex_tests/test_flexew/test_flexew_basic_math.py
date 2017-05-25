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
from ngraph.testing.flexutil import template_one_placeholder, MINIMUM_FLEX_VALUE, \
    MAXIMUM_FLEX_VALUE, id_func, template_two_placeholders, assert_allclose
from ngraph.testing import executor

pytestmark = [pytest.mark.transformer_dependent("module"),
              pytest.mark.flex_only]

# Known issues
bug_1064 = pytest.mark.xfail(strict=True, reason="GitHub issue #1064, flex lower priority issues:"
                                                 "modulus and ZeroDivisionError clarification")
bug_1227 = pytest.mark.xfail(strict=True, reason="GitHub issue #1227, find explanation of results")

test_assign_data = (
    # template: ([(operand, expected_result, *case_description)], test_description),
    # *case_description is optional

    # test_assign
    ([(MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE)], "Assign function - underflow expected"),
    ([(MAXIMUM_FLEX_VALUE + 1, MAXIMUM_FLEX_VALUE)], "Assign function - overflow expected"),
    ([(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE)], "Assign function of negative boundary value"),
    ([(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)], "Assign function of positive boundary value"),
    ([(0.4, 0.399993896484375, "High precision check")],
     "Assign function of positive values from flex range - check high precision"),
    ([(1, 1),
      (5, 1.9999, "Operand overflow to 1.9999"),
      (10, 7.9997, "Operand overflow to 7.9997"),
      (MINIMUM_FLEX_VALUE, -32, "Operand underflow to -32")],
     "Iterations assign of x"),
)

test_data_single_operand = (
    # template: (ng_operation, [(operand, expected_result, *case_description)], test_description),
    # *case_description is optional

    # test_neg
    (ng.negative, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)],
     "Negate function - overflow expected"),
    (ng.negative, [(MINIMUM_FLEX_VALUE + 1, MAXIMUM_FLEX_VALUE)],
     "Negate function of negative boundary value inside of flex range"),
    (ng.negative, [(MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE + 1)],
     "Negate function of positive boundary value inside of flex range"),
    # test_sqrt
    (ng.sqrt, [(0, 0)], "Square root of zero equals zero"),
    (ng.sqrt, [(MAXIMUM_FLEX_VALUE, 181.015625, "High precision check")], "Square root of positive boundary value"),
    (ng.sqrt, [(MINIMUM_FLEX_VALUE, 0, "sqrt is implemented as LUT so input out of range saturates")],
     "Square root of negative boundary value"),
    (ng.sqrt, [(1, 1),
               (124, 1.4141, "Overflow of operand to 1.9999"),
               (10000, 2.8283, "Overflow of operand to 7.9997"),
               (32000, 5.6567, "Overflow of operand to 31.9990")],
     "Iterations sqrt of x "),

    # test_abs
    (ng.absolute, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)],
     "Absolute value from the flex range - overflow expected"),
    (ng.absolute, [(MAXIMUM_FLEX_VALUE - 2, MAXIMUM_FLEX_VALUE - 2)],
     "Absolute value within of the flex range"),
    (ng.absolute, [(-1, 1),
                   (10000, 1.9999, "Overflow of operand to 1.9999"),
                   (-0.4, 0.3999),
                   (MINIMUM_FLEX_VALUE, 15.9995, "Overflow of operand to 15.9995")],
     "Iterations abs of x"),
    (ng.absolute, [(np.array([1, 2, 14, 4, 5, 6, 7, 8, 9, -1]),
                    np.array([1, 2, 14, 4, 5, 6, 7, 8, 9, 1]))],
     "Abs of array"),
    (ng.absolute, [(np.array([5, -1, 0, -2, 3, 4]).reshape(2, 3),
                    (np.array([5, 1, 0, 2, 3, 4]).reshape(2, 3)))], "Abs of 2x3 array"),
)

test_data_double_operand = (
    # template:(ng_operation, [(operand_1, operand_2, expected_result, *case_description)],
    # test_description), *case_description is optional

    # test_add
    (ng.add, [(MAXIMUM_FLEX_VALUE, 2, MAXIMUM_FLEX_VALUE)],
     "Positive boundary value plus one - overflow expected"),
    (ng.add, [(MINIMUM_FLEX_VALUE, 1, MINIMUM_FLEX_VALUE + 1)],
     "Negative boundary value plus one"),
    (ng.add, [(0, 1.5, 1.5),
              (1, 1.5, 1.99993896484375, "Overflow of result to 1.9999"),
              (2, 1.5, 3.5),
              (3, 1.5, 4.5)],
     "Iterations x + 1.5"),
    (ng.add, [(np.array([1, 1, 1, 1]), np.array([2, 3, 4, 5]), np.array([3, 4, 5, 6]))],
     "Add element-wise two arrays to each other"),

    # test_subtraction
    (ng.subtract, [(MINIMUM_FLEX_VALUE, 1, MINIMUM_FLEX_VALUE)],
     "Negative boundary value minus one - underflow expected"),
    (ng.subtract, [(MINIMUM_FLEX_VALUE, 2, MINIMUM_FLEX_VALUE)],
     "Negative boundary value minus two - underflow expected"),
    (ng.subtract, [(MAXIMUM_FLEX_VALUE, 1, MAXIMUM_FLEX_VALUE - 1)],
     "Positive boundary value minus one"),
    (ng.subtract, [(MAXIMUM_FLEX_VALUE, 2, MAXIMUM_FLEX_VALUE - 2)],
     "Positive boundary value minus two"),
    (ng.subtract, [(10, 0.4, 9.6),
                   (1000, 0.4, 15.5996, "Overflow of operand 1 to 15.9995"),
                   (10000, 0.4, 31.999, "Overflow of operand 1 to 63.9980"),
                   (1, 0.4, 0.6015)],
     "Iterations x - 0.4"),
    (ng.subtract, [(10000, 0.4, 9999.5, "Overflow of result to 9999.5"),
                   (5000, 0.4, 4999.5, "Overflow of result to 4999.5 "),
                   (2500, 0.4, 2500, "Overflow of result to 2500"),
                   (1000, 0.4, 1000, "Overflow of result to 1000"),
                   (800, 0.4, 800, "Overflow of result to 800"),
                   (750, 0.4, 750, "Overflow of result to 750"),
                   (600, 0.4, 600, "Overflow of result to 600"),
                   (500, 0.4, 500, "Overflow of result to 500"),
                   (300, 0.4, 300, "Overflow of result to 300"),
                   (200, 0.4, 200, "Overflow of result to 200"),
                   (250, 0.4, 250, "Overflow of result to 250"),
                   (100, 0.4, 100, "Overflow of result to 100"),
                   (80, 0.4, 80, "Overflow of result to 80"),
                   (75, 0.4, 75, "Overflow of result to 75"),
                   (60, 0.4, 60, "Overflow of result to 60"),
                   (50, 0.4, 50, "Overflow of result to 50"),
                   (30, 0.4, 30, "Overflow of result to 30"),
                   (25, 0.4, 24.5, "Overflow of result to 24.5"),
                   (20, 0.4, 19.5, "Overflow of result to 20"),
                   (10, 0.4, 9.625, "Overflow of result to 10"),
                   (1, 0.4, 0.625, "Overflow of result to 1")],
     "More complex iterations x - 0.4"),
    (ng.subtract, [(np.array([2, 3, 4, 5]), np.array([1, 1, 1, 1]), np.array([1, 2, 3, 4]))],
     "Subtract element-wise two arrays to each other"),

    # test_multiplication
    (ng.multiply, [(MINIMUM_FLEX_VALUE, 2, MINIMUM_FLEX_VALUE)],
     "Negative boundary value multiplied by two - underflow expected",),
    (ng.multiply, [(MAXIMUM_FLEX_VALUE, 2, MAXIMUM_FLEX_VALUE)],
     "Positive boundary value multiplied by two - overflow expected",),
    (ng.multiply, [(MINIMUM_FLEX_VALUE, 0, 0)],
     "Negative boundary value multiplied by zero equals zero"),
    (ng.multiply, [(MAXIMUM_FLEX_VALUE, 1, MAXIMUM_FLEX_VALUE)],
     "Positive boundary value multiplied by one is the same"),
    (ng.multiply, [(0, 0, 0)], "Zero multiplied by zero equals zero"),
    (ng.multiply, [(MINIMUM_FLEX_VALUE, -0.5, MINIMUM_FLEX_VALUE * (-0.5))],
     "Negative sign value multiplied by negative sign value equals positive sign"),
    (ng.multiply, [(1, 10.1, 10.0996),
                   (1000, 10.1, 15.9995, "Overflow of operand 1 to 1.9999 and result to 15.999"),
                   (2, 10.1, 20.1992,),
                   (-1000, 10.1, -64, "Underflow of operand 1 to -8 and result to -64"),
                   (0.4, 10.1, 4.0312)],
     "Iterations x * 10.1"),
    (ng.multiply, [(np.array([2, 3, 4, 5]), np.array([1, 1, 1, 1]), np.array([2, 3, 4, 5]))],
     "Multiply element-wise two arrays to each other"),

    # test_division
    (ng.divide, [(MAXIMUM_FLEX_VALUE, 0.5, MAXIMUM_FLEX_VALUE)],
     "Positive boundary value division - overflow expected"),
    (ng.divide, [(MINIMUM_FLEX_VALUE, 0.5, MINIMUM_FLEX_VALUE)],
     "Negative boundary value division - underflow expected"),
    bug_1227((ng.divide, [(MAXIMUM_FLEX_VALUE, 3, 10922)], "Positive boundary value division")),
    bug_1227((ng.divide, [(MINIMUM_FLEX_VALUE, 3, -10922)], "Negative boundary value division")),
    (ng.divide, [(-10, 7, -1.4285),
                 (0.4, 7, 0.0571),
                 (MAXIMUM_FLEX_VALUE, 7, 3.9998,
                  "Overflow operand 1 to 31.9990 and result to 3,9998"),
                 (MINIMUM_FLEX_VALUE, 7, -16, "Underflow operand 1 to -128 and result to -16"),
                 (0, 7, 0)],
     "Iterations x / 7"),

    # test_modulo
    bug_1064((ng.mod, [(MINIMUM_FLEX_VALUE, 3, MINIMUM_FLEX_VALUE % 3)],
              "Negative boundary value mod 3")),
    (ng.mod, [(MAXIMUM_FLEX_VALUE, 3, MAXIMUM_FLEX_VALUE % 3)], "Positive boundary value mod 3"),
    bug_1064((ng.mod, [(MAXIMUM_FLEX_VALUE, -3, MAXIMUM_FLEX_VALUE % (-3))],
              "Positive boundary value mod -3")),
    bug_1064((ng.mod, [(2.1, 2, 0.09999847412109375, "High precision check")],
              "Modulo of floating point")),

    # test_power
    (ng.power, [(MAXIMUM_FLEX_VALUE, 2, MAXIMUM_FLEX_VALUE)],
     "Positive boundary value exponentiation - overflow expected"),
    (ng.power, [(MINIMUM_FLEX_VALUE, 3, MINIMUM_FLEX_VALUE)],
     "Negative boundary value exponentiation - underflow expected"),
    # Not sure of this case, results should be tracked
    (ng.power, [(MAXIMUM_FLEX_VALUE, 0.4, 63.99609375, "High precision check")],
     "Positive boundary value exponentiation"),
    (ng.power, [(MINIMUM_FLEX_VALUE, -2, MINIMUM_FLEX_VALUE ** (-2))],
     "Negative boundary value negative exponentiation"),
    (ng.power, [(1, 3, 1),
                (5, 3, 1.9999, "Underflow operand 1 and result to 1.9999"),
                (-10, 3, -8, "Underflow operand 1 and result to -8"),
                (22, 3, 31.999, "Overflow result to 31.999")],
     "Iterations x ^ 3")
)


@pytest.mark.parametrize("operation, operands, test_name", test_data_single_operand, ids=id_func)
def test_single_operand(transformer_factory, operation, operands, test_name):
    template_one_placeholder(operands, operation)


@pytest.mark.parametrize("operation, operands, test_name", test_data_double_operand, ids=id_func)
def test_double_operand(transformer_factory, operation, operands, test_name):
    template_two_placeholders(operands, operation)


@pytest.mark.parametrize("operands, test_name", test_assign_data, ids=id_func)
def test_assign(transformer_factory, operands, test_name):
    v = ng.variable(())
    ng_placeholder = ng.placeholder(())
    vset = ng.sequential([
        ng.assign(v, ng_placeholder),
        v
    ])
    iterations = len(operands) != 1
    with executor(vset, ng_placeholder) as ex:
        for i in operands:
            flex_result = ex(i[0])
            print("flex: ", flex_result)
            print("expected: ", i[1])
            if iterations:
                assert_allclose(flex_result, i[1])
            else:
                assert flex_result == i[1]
