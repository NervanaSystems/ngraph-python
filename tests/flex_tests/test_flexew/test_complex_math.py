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
from ngraph.testing.flex_util import template_one_placeholder, id_func, MINIMUM_FLEX_VALUE, \
    MAXIMUM_FLEX_VALUE

pytestmark = pytest.mark.flex_only

test_data_single_operand = (
    # template: (ng_operation, [(operand, expected_result, *case_description)], test_description),
    # *case_description is optional

    # test_exp
    (ng.exp, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)],
     "Exponential function - overflow expected"),
    (ng.exp, [(1.0, 2.71826171875, "High precision check")], "Exponential function of 1"),
    (ng.exp, [(0.0, 1.0)], "Exponential function of 0"),
    (ng.exp, [(int(np.log(MAXIMUM_FLEX_VALUE) / 2), 148.4140625)],
     "Exponential function of a positive value inside of flex range"),
    (ng.exp, [(-int(np.log(MAXIMUM_FLEX_VALUE)) / 2, 0.0067379474639892578,
               "High precision check")],
     "Exponential function of a negative value inside of flex range"),
    (ng.exp, [(0, 1),
              (10, 1.9999, "overflow of operand to 1.9999"),
              (1, 2.7182),
              (-5, 0.0068)],
     "Iterations exp of x"),

    # test_log
    (ng.log, [(0, MINIMUM_FLEX_VALUE, "-INF leads to underflow")],
     "Logarithm of 0 to achieve underflow, -inf"),
    (ng.log, [(0.01, -4.605224609375, "High precision check")],
     "Logarithm of a small constant within the flex range"),
    (ng.log, [(1, np.log(1))], "Logarithm of 1 to achieve 0"),
    (ng.log, [(MAXIMUM_FLEX_VALUE, 10.39697265625, "High precision check")],
     "Logarithm of a positive border value to achieve a number from flex range"),
    (ng.log, [(MAXIMUM_FLEX_VALUE + 1, 10.39697265625, "High precision check")],
     "Logarithm of a value greater than positive border value to achieve overflow of the operand"),
    (ng.log, [(2, 0.6931),
              (MAXIMUM_FLEX_VALUE, 0.9999),
              (10, 2.3026),
              (1, 0)],
     "Iterations ln of x"),

    # test_safe_log
    (ng.safelog, [(0, -50.0, "Limited by safelog to -50")], "Safe-Logarithm of 0 to limit -50"),
    (ng.safelog, [(0.01, -4.605224609375, "High precision check")],
     "Safe-Logarithm of a small constant within the flex range"),
    (ng.safelog, [(1, np.log(1))], "Safe-Logarithm of 1 to achieve 0"),
    (ng.safelog, [(MAXIMUM_FLEX_VALUE, 10.39697265625)],
     "Safe-Logarithm of a positive border value to achieve a number from flex range"),
    (ng.safelog, [(MAXIMUM_FLEX_VALUE + 1, 10.39697265625)],
     "Safe-Logarithm of a value greater than positive border value to achieve overflow"),

    # test_tanh
    (ng.tanh, [(MINIMUM_FLEX_VALUE - 1, np.tanh(MINIMUM_FLEX_VALUE))],
     "Tanh of a constant below the flex range"),
    (ng.tanh, [(MINIMUM_FLEX_VALUE, np.tanh(MINIMUM_FLEX_VALUE))],
     "Tanh of a negative border value"),
    (ng.tanh, [(0, np.tanh(0))], "Tanh of 0"),
    (ng.tanh, [(MAXIMUM_FLEX_VALUE, np.tanh(MAXIMUM_FLEX_VALUE))],
     "Tanh of a positive border value"),
    (ng.tanh, [(MAXIMUM_FLEX_VALUE + 1, np.tanh(MAXIMUM_FLEX_VALUE))],
     "Tanh of a constant above the flex range"),

    # test_reciprocal
    (ng.reciprocal, [(1, 1,)], "Reciprocal of 1 equals 1"),
    (ng.reciprocal, [(MINIMUM_FLEX_VALUE, -3.0517578125e-05, "High precision check")],
     "Reciprocal of negative boundary value"),
    (ng.reciprocal, [(1, 1),
                     (MAXIMUM_FLEX_VALUE, 0.5, "Overflow of operand to 1.9999"),
                     (2, 0.5),
                     (MINIMUM_FLEX_VALUE, -0.125, "Underflow of operand to -8")],
     "Iterations reciprocal of x"),

    #  test_square
    (ng.square, [(1, 1)], "Square of 1 equals 1"),
    (ng.square, [(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)],
     "Square of negative boundary value - overflow expected"),
    (ng.square, [(MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE)],
     "Square of positive boundary value - overflow expected"),

    # test_sigmoid and sigmoidAtomic
    (ng.sigmoid, [(-10, 0.00004539825022220611572265625)], ""),
    (ng.sigmoidAtomic, [(-10, 0.00004539825022220611572265625)], ""),
    (ng.sigmoid, [(-11, 0.000030517578125)],
     "denominator in sigmoid equation is saturating to MAX_FLEX_VALUE"),
    (ng.sigmoidAtomic, [(-11, 0.000016701407730579376220703125)], ""),
    (ng.sigmoid, [(-20, 0.000030517578125)],
     "denominator in sigmoid equation is saturating to MAX_FLEX_VALUE"),
    (ng.sigmoidAtomic, [(-20, 0.00000000186264514923095703125)], ""),
    (ng.sigmoid, [(-30, 0.000030517578125)],
     "denominator in sigmoid equation is saturating to MAX_FLEX_VALUE"),
    (ng.sigmoidAtomic, [(-30, 0.0)], ""),
)

ExpMinus11 = 0.000016701407730579376220703125
Minus11 = -11
test_input_types_cases = (
    # template:(operation, operand, expected_result, description, placeholder)
    (ng.exp, [(Minus11, ExpMinus11)], "Exponential function of -11 as scalar"),
    (ng.exp, [(np.array([Minus11]), np.array([ExpMinus11]))],
     "Exponential function of -11 as 1-element array"),
    (ng.exp, [(np.array([Minus11, Minus11]), np.array([ExpMinus11, ExpMinus11]))],
     "Exponential function of -11 as multi-element array"),
)


@pytest.mark.parametrize("operation, operands, test_name", test_data_single_operand, ids=id_func)
def test_single_operand(transformer_factory, operation, operands, test_name):
    template_one_placeholder(operands, operation)


@pytest.mark.parametrize("operation, operands, test_name", test_input_types_cases, ids=id_func)
def test_input_types(transformer_factory, operation, operands, test_name):
    template_one_placeholder(operands, operation)
