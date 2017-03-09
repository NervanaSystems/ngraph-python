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
    # template:(operation, operand, expected_result, description

    # test_assign
    bug((op.pos, MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE, "Assign function - underflow expected")),
    bug((op.pos, MAXIMUM_FLEX_VALUE + 1, MAXIMUM_FLEX_VALUE, "Assign function - overflow expected")),
    (op.pos, MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, "Assign function of negative boundary value"),
    (op.pos, MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, "Assign function of positive boundary value"),

    # test_neg
    bug((op.neg, MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, "Negate function - overflow expected")),
    bug((op.neg, MINIMUM_FLEX_VALUE + 1, MAXIMUM_FLEX_VALUE,
        "Assign function of negative boundary value inside of flex range")),
    bug((op.neg, MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE + 1,
        "Assign function of positive boundary value inside of flex range")),

    # test_sqrt
    (ng.sqrt, 0, 0, "Square root of zero i zero"),
    (ng.sqrt, MAXIMUM_FLEX_VALUE, np.sqrt(MAXIMUM_FLEX_VALUE), "Square root of positive boundary value"),
    bug((ng.sqrt, MINIMUM_FLEX_VALUE, np.sqrt(MINIMUM_FLEX_VALUE), "Square of negative boundary value - NaN expected")),

    # test_abs
    bug((ng.absolute, MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE,
         "Absolute value from the flex range - overflow expected")),
    bug((ng.absolute, MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, "Absolute value outside of the flex range"))
)

test_data_double_operand = (
    # template:(operation, operand_1, operand_2, expected_result, description

    # test_add
    bug((op.add, MAXIMUM_FLEX_VALUE, 1,  MAXIMUM_FLEX_VALUE, "Positive boundary value plus one - overflow expected")),
    bug((op.add, MINIMUM_FLEX_VALUE, 1, MINIMUM_FLEX_VALUE + 1, "Negative boundary value plus one")),

    # test_subtraction
    (op.sub, MINIMUM_FLEX_VALUE, 1,  MINIMUM_FLEX_VALUE, "Negative boundary value minus one - underflow expected"),
    bug((op.sub, MINIMUM_FLEX_VALUE, 2,  MINIMUM_FLEX_VALUE, "Negative boundary value minus two - underflow expected")),
    (op.sub, MAXIMUM_FLEX_VALUE, 1,  MAXIMUM_FLEX_VALUE - 1, "Positive boundary value minus one"),
    bug((op.sub, MAXIMUM_FLEX_VALUE, 2,  MAXIMUM_FLEX_VALUE - 2, "Positive boundary value minus two")),

    # test_multiplication
    bug((op.mul, MINIMUM_FLEX_VALUE, 2, MINIMUM_FLEX_VALUE, "Negative boundary value by two - underflow expected")),
    bug((op.mul, MAXIMUM_FLEX_VALUE, 2, MAXIMUM_FLEX_VALUE, "Positive boundary value by two - overflow expected")),
    (op.mul, MINIMUM_FLEX_VALUE, 0, 0, "Negative boundary value by zero equals zero"),
    (op.mul, MAXIMUM_FLEX_VALUE, 1, MAXIMUM_FLEX_VALUE, "Positive boundary value by one is the same"),
    (op.mul, 0, 0, 0, "Zero by zero should equal to zero"),
    (op.mul, MINIMUM_FLEX_VALUE, -0.5, MINIMUM_FLEX_VALUE * (-0.5),
     "Negative sign by negative sin equals positive sign"),

    # test_division
    bug((op.div, MAXIMUM_FLEX_VALUE, 0.5, MAXIMUM_FLEX_VALUE, "Positive boundary value division - overflow expected")),
    bug((op.div, MINIMUM_FLEX_VALUE, 0.5, MINIMUM_FLEX_VALUE, "Negative boundary value division - underflow expected")),
    bug((op.div, MAXIMUM_FLEX_VALUE, 3, MAXIMUM_FLEX_VALUE / 3, "Positive boundary value division")),
    bug((op.div, MINIMUM_FLEX_VALUE, 3, MINIMUM_FLEX_VALUE / 3, "Negative boundary value division")),

    # test_modulo
    bug((op.mod, MINIMUM_FLEX_VALUE, 3, MINIMUM_FLEX_VALUE % 3, "Negative boundary value mod 3")),
    (op.mod, MAXIMUM_FLEX_VALUE, 3, MAXIMUM_FLEX_VALUE % 3, "Positive boundary value mod 3"),
    bug((op.mod, MAXIMUM_FLEX_VALUE, -3, MAXIMUM_FLEX_VALUE % (-3), "Positive boundary value mod -3")),

    # test_power
    bug((op.pow, MAXIMUM_FLEX_VALUE, 2, MAXIMUM_FLEX_VALUE,
         "Positive boundary value exponentiation - overflow expected")),
    bug((op.pow, MINIMUM_FLEX_VALUE, 3, MINIMUM_FLEX_VALUE,
        "Negative boundary value exponentiation - underflow expected")),
    (op.pow, MAXIMUM_FLEX_VALUE, 0.5, MAXIMUM_FLEX_VALUE ** 0.5, "Positive boundary value exponentiation"),
    (op.pow, MINIMUM_FLEX_VALUE, -2, MINIMUM_FLEX_VALUE ** (-2), "Negative boundary value negative exponentiation")
)


@pytest.mark.parametrize("operation, operand, expected_result, ,  description", test_data_single_operand)
def test_single_operand(transformer_factory, operation, operand, expected_result, description):
    template_one_placeholder(operand, operation(x), x, expected_result, description)


@pytest.mark.parametrize("operation, operand_1, operand_2, expected_result, description", test_data_double_operand)
def test_single_operand(transformer_factory, operation, operand_1, operand_2, expected_result, description):
    template_one_placeholder(operand_1, operation(x, operand_2), x, expected_result, description)
