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
import ngraph as ng
from ngraph.testing.flexutil import template_one_placeholder, id_func

pytestmark = pytest.mark.transformer_dependent("module")

MINIMUM_FLEX_VALUE = -2 ** 15
MAXIMUM_FLEX_VALUE = 2 ** 15 - 1

x = ng.placeholder(())

test_data_single_operand = (
    # template:(operation, operand, expected_result, description)

    # test_exp
    (ng.exp, [MAXIMUM_FLEX_VALUE], [MAXIMUM_FLEX_VALUE], "Exponential function - overflow expected"),
    (ng.exp, [1.0], [2.71826171875], "Exponential function of 1"),
    (ng.exp, [0.0], [1.0], "Exponential function of 0"),
    (ng.exp, [int(np.log(MAXIMUM_FLEX_VALUE) / 2)], [148.4140625],
     "Exponential function of a positive value inside of flex range"),
    (ng.exp, [-int(np.log(MAXIMUM_FLEX_VALUE) / 2)], [0.0067379474639892578],
     "Exponential function of a negative value inside of flex range"),

    # test_log
    (ng.log, [0], [MINIMUM_FLEX_VALUE], "Logarithm of 0 to achieve underflow (-inf)"),
    (ng.log, [0.01], [-4.605224609375], "Logarithm of a small constant within the flex range"),
    (ng.log, [1], [np.log(1)], "Logarithm of 1 to achieve 0"),
    (ng.log, [MAXIMUM_FLEX_VALUE], [10.39697265625],
     "Logarithm of a positive border value to achieve a number from flex range"),
    (ng.log, [MAXIMUM_FLEX_VALUE + 1], [10.39697265625],
     "Logarithm of a value greater than positive border value to achieve overflow"),

    # test_safe_log
    (ng.safelog, [0], [-50.0], "Safe-Logarithm of 0 to limit (-50)"),
    (ng.safelog, [0.01], [-4.605224609375], "Safe-Logarithm of a small constant within the flex range"),
    (ng.safelog, [1], [np.log(1)], "Safe-Logarithm of 1 to achieve 0"),
    (ng.safelog, [MAXIMUM_FLEX_VALUE], [10.39697265625],
     "Safe-Logarithm of a positive border value to achieve a number from flex range"),
    (ng.safelog, [MAXIMUM_FLEX_VALUE + 1], [10.39697265625],
     "Safe-Logarithm of a value greater than positive border value to achieve overflow"),

    # test_tanh
    (ng.tanh, [MINIMUM_FLEX_VALUE - 1], [np.tanh(MINIMUM_FLEX_VALUE)], "Tanh of a constant below the flex range"),
    (ng.tanh, [MINIMUM_FLEX_VALUE], [np.tanh(MINIMUM_FLEX_VALUE)], "Tanh of a negative border value"),
    (ng.tanh, [0], [np.tanh(0)], "Tanh of 0"),
    (ng.tanh, [MAXIMUM_FLEX_VALUE], [np.tanh(MAXIMUM_FLEX_VALUE)], "Tanh of a positive border value"),
    (ng.tanh, [MAXIMUM_FLEX_VALUE + 1], [np.tanh(MAXIMUM_FLEX_VALUE)], "Tanh of a constant above the flex range"),

    # test_reciprocal
    (ng.reciprocal, [1], [1], "Reciprocal of 1 equals 1"),
    (ng.reciprocal, [MAXIMUM_FLEX_VALUE], [3.0517578125e-05], "Reciprocal of positive boundary value - high precision"),
    (ng.reciprocal, [MINIMUM_FLEX_VALUE], [-3.0517578125e-05],
     "Reciprocal of negative boundary value - high precision"),

    # test_square
    (ng.square, [1], [1], "Square of 1 equals 1"),
    (ng.square, [MINIMUM_FLEX_VALUE], [MAXIMUM_FLEX_VALUE],
              "Square of negative boundary value - overflow expected"),
    (ng.square, [MAXIMUM_FLEX_VALUE], [MAXIMUM_FLEX_VALUE],
              "Square of positive boundary value - overflow expected")
)


ExpMinus11 = 0.000016701407730579376220703125
Minus11 = -11
test_input_types_cases = (
    # template:(operation, operand, expected_result, description, placeholder)
    (ng.exp, [Minus11], [ExpMinus11],
     "Exponential function of -11 as scalar",
     ng.placeholder(())),

    (ng.exp, [np.array([Minus11])], np.array([ExpMinus11]),
     "Exponential function of -11 as 1-element array",
     ng.placeholder(ng.make_axes([ng.make_axis(length=1)]))),

    (ng.exp, [np.array([Minus11, Minus11])], [np.array([ExpMinus11, ExpMinus11])],
     "Exponential function of -11 as multi-element array",
     ng.placeholder(ng.make_axes([ng.make_axis(length=2)]))),
)


@pytest.mark.parametrize("operation, operand, expected_result, description", test_data_single_operand, ids=id_func)
def test_single_operand(transformer_factory, operation, operand, expected_result, description):
    template_one_placeholder(operand, operation(x), x, expected_result, description)


@pytest.mark.parametrize("operation, operand, expected_result, description, placeholder", test_input_types_cases, ids=id_func)
def test_input_types(transformer_factory, operation, operand, expected_result, description, placeholder):
    template_one_placeholder(operand, operation(placeholder), placeholder, expected_result, description)







