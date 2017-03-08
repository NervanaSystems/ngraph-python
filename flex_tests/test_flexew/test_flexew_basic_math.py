import numpy as np
import pytest

import ngraph as ng
from ngraph.testing import executor, template_one_placeholder


MINIMUM_FLEX_VALUE = -2**15
MAXIMUM_FLEX_VALUE = 2**15 - 1

EPSILON = 0.2
x = ng.placeholder(())
z = ng.placeholder(())


# Assignment
test_data = {
    "test_assign": [
        pytest.mark.xfail((MINIMUM_FLEX_VALUE - 2, MINIMUM_FLEX_VALUE,
                           "Assign function - underflow expected"), strict=True),
        pytest.mark.xfail((MAXIMUM_FLEX_VALUE + 1, MAXIMUM_FLEX_VALUE,
                           "Assign function - overflow expected"), strict=True),
        (MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, "Assign function of negative boundary value inside of flex range"),
        (MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, "Assign function of positive boundary value inside of flex range")
    ],
    "test_negate": [
        pytest.mark.xfail((MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE,
                           "Negate function - overflow expected"), strict=True),
        pytest.mark.xfail((MINIMUM_FLEX_VALUE + 1, MAXIMUM_FLEX_VALUE,
                           "Assign function of negative boundary value inside of flex range"), strict=True),
        pytest.mark.xfail((MAXIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE + 1,
                           "Assign function of positive boundary value inside of flex range"), strict=True)
    ],
    "test_absolute": [
        pytest.mark.xfail((MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE,
                           "Absolute value from the flex range - overflow expected"), strict=True),
        pytest.mark.xfail((MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE,
                           "Absolute value outside of the flex range"), strict=True)
     ],
    "test_addition": [
        pytest.mark.xfail((MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE,
                           "Positive boundary value plus one - overflow expected"), strict=True),
        pytest.mark.xfail((MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE + 1,
                           "Negative boundary value plus one"), strict=True)
    ],
    "test_subtraction": [
        # template:(minuend, subtrahend, expected_difference, description)
        (MINIMUM_FLEX_VALUE, 1,  MINIMUM_FLEX_VALUE, "Negative boundary value minus one - underflow expected"),
        pytest.mark.xfail((MINIMUM_FLEX_VALUE, 2,  MINIMUM_FLEX_VALUE,
                           "Negative boundary value minus two - underflow expected"), strict=True),
        (MAXIMUM_FLEX_VALUE, 1,  MAXIMUM_FLEX_VALUE - 1, "Positive boundary value minus one"),
        pytest.mark.xfail((MAXIMUM_FLEX_VALUE, 2,  MAXIMUM_FLEX_VALUE - 2,
                           "Positive boundary value minus two"), strict=True)
    ],
    "test_multiplication": [
        # template:(multiplier_1, multiplier_2, expected_product, description)
        pytest.mark.xfail((MINIMUM_FLEX_VALUE, 2, MINIMUM_FLEX_VALUE,
                           "Negative boundary value by two - underflow expected"), strict=True),
        pytest.mark.xfail((MAXIMUM_FLEX_VALUE, 2, MAXIMUM_FLEX_VALUE,
                           "Positive boundary value by two - overflow expected"), strict=True),
        (MINIMUM_FLEX_VALUE, 0, 0, "Negative boundary value by zero equals zero"),
        (MAXIMUM_FLEX_VALUE, 1, MAXIMUM_FLEX_VALUE, "Positive boundary value by one is the same")
    ]
}


@pytest.mark.parametrize("test_input, expected, description", test_data["test_assign"])
def test_assign(transformer_factory, test_input, expected, description):
    template_one_placeholder(test_input, x, x, expected, description)


@pytest.mark.parametrize("test_input, expected, description", test_data["test_negate"])
def test_negate(transformer_factory, test_input, expected, description):
    template_one_placeholder(test_input, -x, x, expected, description)


@pytest.mark.parametrize("test_input, expected, description", test_data["test_absolute"])
def test_absolute(transformer_factory, test_input, expected, description):
    template_one_placeholder(test_input, ng.absolute(x), x, expected, description)


@pytest.mark.parametrize("test_input, expected, description", test_data["test_addition"])
def test_addition(transformer_factory, test_input, expected, description):
    template_one_placeholder(test_input, x + 1, x, expected, description)


@pytest.mark.parametrize("minuend, subtrahend, expected_difference, description", test_data["test_subtraction"])
def test_subtraction(transformer_factory, minuend, subtrahend, expected_difference, description):
    template_one_placeholder(minuend, x - subtrahend, x, expected_difference, description)


@pytest.mark.parametrize("multiplier_1, multiplier_2, expected_product, description", test_data["test_multiplication"])
def test_multiplication(transformer_factory, multiplier_1, multiplier_2, expected_product, description):
    template_one_placeholder(multiplier_1, x * multiplier_2, x, expected_product, description)


# # Multiplication.
# def test_multiplication_const_minimum_flex_range_value(transformer_factory):
#     """
#     Multiplies by a constant to achieve a number below the flex range.
#     Negative case.
#     """
#     template_one_placeholder([MINIMUM_FLEX_VALUE - 1], x, x * 3.3, lambda y: y * 3.3, expect_error=True)
#
#
# def test_multiplication_negative_const_inside_flex_range_value(transformer_factory):
#     """
#     Multiplies by a negative constant to achieve a number in between the flex range.
#     """
#     template_one_placeholder(np.arange(int(MINIMUM_FLEX_VALUE / 4), int(MAXIMUM_FLEX_VALUE / 4), 0.1), x, x * -3.3,
#                              lambda y: y * -3.3)
#
#
# def test_multiplication_positive_const_inside_flex_range_value(transformer_factory):
#     """
#     Multiplies by a positive constant to achieve a number in between the flex range.
#     """
#     template_one_placeholder(np.arange(int(MINIMUM_FLEX_VALUE / 4), int(MAXIMUM_FLEX_VALUE / 4), 0.1), x, x * 3.3,
#                              lambda y: y * 3.3)
#
#
# def test_multiplication_zero_inside_flex_range_value(transformer_factory):
#     """
#     Multiplies by 0 to achieve 0.
#     """
#     template_one_placeholder(np.arange(int(MINIMUM_FLEX_VALUE), int(MAXIMUM_FLEX_VALUE), 0.1), x, x * 0, lambda y: 0)
#
#
# def test_multiplication_const_above_maximum_flex_range_value(transformer_factory):
#     """
#     Multiplies by a constant to achieve a number above the flex range.
#     Negative case.
#     """
#     template_one_placeholder([MAXIMUM_FLEX_VALUE / 2], x, x * 3.3, lambda y: y * 3.3, expect_error=True)
#
#
# # Division
# def test_division_const_below_minimum_flex_range_value(transformer_factory):
#     """
#     Divides by a constant to achieve a number below the flex range.
#     Negative case.
#     """
#     template_one_placeholder([MINIMUM_FLEX_VALUE], x, x / 0.3, lambda y: y / 0.3, expect_error=True)
#
#
# def test_division_negative_const_inside_flex_range_value(transformer_factory):
#     """
#     Divides by a negative constant to achieve a number between the flex range.
#     """
#     template_one_placeholder(np.arange(MINIMUM_FLEX_VALUE / 2, MAXIMUM_FLEX_VALUE / 2, 0.1), x, x / -1.5,
#                              lambda y: y / -1.5)
#
#
# def test_division_positive_const_inside_flex_range_value(transformer_factory):
#     """
#     Divides by a positive constant to achieve a number in between the flex range.
#     """
#     template_one_placeholder(np.arange(MINIMUM_FLEX_VALUE / 2, MAXIMUM_FLEX_VALUE / 2, 0.1), x, x / 1.5,
#                              lambda y: y / 1.5)
#
#
# @pytest.mark.xfail(reason="To fix later")
# def test_division_zero_inside_flex_range_value(transformer_factory):
#     """
#     Multiplies by 0 to achieve an error.
#     Negative case.
#     ZeroDivisionError doesn't occur - should be fixed in the future
#     """
#     x_multiplied_by_constant = x / 0
#     with pytest.raises(ZeroDivisionError):
#         with executor(x_multiplied_by_constant, x) as mulconst_executor:
#             for _value in range(int(MINIMUM_FLEX_VALUE), int(MAXIMUM_FLEX_VALUE)):
#                 # print(mulconst_executor(_value))
#                 mulconst_executor(_value)
#
#
# def test_division_const_above_maximum_flex_range_value(transformer_factory):
#     """
#     Divides by a constant to achieve a number above the flex range.
#     Negative case.
#     """
#     template_one_placeholder([MAXIMUM_FLEX_VALUE / 2], x, x / 0.3, lambda y: y / 0.3, expect_error=True)
#
#
# # Modulo.
# # @pytest.mark.xfail(reason="To fix later")
# def test_modulo_const_below_flex_range_value(transformer_factory):
#     """
#     Modulo of the number below the flex range.
#     Negative case.
#     """
#     template_one_placeholder_equality([(MINIMUM_FLEX_VALUE - 1)], x, x % 2, lambda y: y % 2)
#
#
#
# # @pytest.mark.xfail(reason="To fix later")
# def test_modulo_const_minimum_flex_range_value(transformer_factory):
#     """
#     Modulo of the minimum value of the flex range.
#     """
#     template_one_placeholder_equality((MINIMUM_FLEX_VALUE,), x, x % 2, lambda y: y % 2)
#
#
# # @pytest.mark.xfail(reason="To fix later")
# def test_modulo_negative_const_inside_flex_range_value(transformer_factory):
#     """
#     Modulo negative constant to achieve a number in between the flex range.
#     Positive case.
#     """
#     template_one_placeholder(np.arange(int(MINIMUM_FLEX_VALUE / 2), int(MAXIMUM_FLEX_VALUE / 2), 1), x, x % -2,
#                              lambda y: y % -2)
#
#
#
# # @pytest.mark.xfail(reason="To fix later")
# def test_modulo_const_above_flex_range(transformer_factory):
#     """
#     Modulo of the maximum value of the flex range.
#     Negative case.
#     """
#     template_one_placeholder_equality((MAXIMUM_FLEX_VALUE + 1,), x, x % 2, lambda y: y % 2, expect_error=True)
#
#
# @pytest.mark.xfail(reason="To fix later - modulo of float?")
# def test_modulo_const_above_flex_range(transformer_factory):
#     """
#     Modulo of the maximum value of the flex range.
#     Negative case.
#     """
#     template_one_placeholder_equality((MAXIMUM_FLEX_VALUE,), x, x % 2, lambda y: y % 2)
#
# # Power
# def test_pow_const_below_minimum_flex_range_value(transformer_factory):
#     """
#     Power of a constant to achieve a number below the flex range.
#     Negative case.
#     """
#     template_one_placeholder([MINIMUM_FLEX_VALUE], x, x ** 3, lambda y: y ** 3, expect_error=True)
#
#
# def test_pow_negative_const_inside_flex_range_value(transformer_factory):
#     """
#     Power of a negative constant to achieve a number within the flex range.
#     """
#     template_one_placeholder(np.arange(abs(MINIMUM_FLEX_VALUE) ** (1 / 3.0), MAXIMUM_FLEX_VALUE ** (1 / 3.0), 0.1),
#                              x, x ** -3, lambda y: y ** -3)
#
#
# def test_pow_zero_inside_flex_range_value(transformer_factory):
#     """
#     Power of zero. Should equal to 1 when values are from the inside of flex range.
#     """
#     template_one_placeholder(np.arange(abs(MINIMUM_FLEX_VALUE) ** (1 / 3.0), MAXIMUM_FLEX_VALUE ** (1 / 3.0), 0.1),
#                              x, x ** 0, lambda y: 1)
#
#
# def test_pow_positive_const_inside_flex_range_value(transformer_factory):
#     """
#     Power of a positive constant to achieve a number inside the flex range.
#     """
#     template_one_placeholder(np.arange(abs(MINIMUM_FLEX_VALUE) ** (1 / 3.0), MAXIMUM_FLEX_VALUE ** (1 / 3.0), 0.1),
#                              x, x ** 3, lambda y: y ** 3)
#
#
# def test_pow_const_above_maximum_flex_range_value(transformer_factory):
#     """
#     Power of a constant to achieve a number below the flex range.
#     Negative case.
#     """
#     template_one_placeholder([MAXIMUM_FLEX_VALUE], x, x ** 3, lambda y: y ** 3, expect_error=True)
#
#
# # Square
# def test_sqr_negative_const_inside_flex_range_value(transformer_factory):
#     """
#     Constant squared to achieve a number in between the flex range.
#     """
#     template_one_placeholder(np.arange(abs(MINIMUM_FLEX_VALUE) ** (1 / 2.0), MAXIMUM_FLEX_VALUE ** (1 / 2.0), 0.1),
#                              x, x ** 2, lambda y: y ** 2)
#
#
# def test_sqr_const_above_maximum_flex_range_value(transformer_factory):
#     """
#     Constant squared to achieve a number above the flex range.
#     Negative case.
#     """
#     template_one_placeholder([MAXIMUM_FLEX_VALUE / 2], x, x ** 2, lambda y: y ** 2, expect_error=True)
#
#
# # Square root
#
# @pytest.mark.xfail(reason="Sqrt of negative number is not NaN - Known Issue")
# def test_sqrt_negative_const_inside_flex_range_value(transformer_factory):
#     """
#     Square root of a negative constant. Should fail.
#     Negative case.
#     """
#     with executor(ng.sqrt(x), x) as sqrt_executor:
#         assert np.isnan(sqrt_executor(MINIMUM_FLEX_VALUE/2))
#
#
#
# def test_sqrt_zero_inside_flex_range_value(transformer_factory):
#     """
#     Square root of a zero.
#     """
#     template_one_placeholder([0], x, ng.sqrt(x), lambda y: np.sqrt(y))
#
#
#
# # @pytest.mark.xfail(reason="Sqrt of positive value get 0 instead of proper value")
# def test_sqrt_positive_const_inside_flex_range_value(transformer_factory):
#     """
#     Square root of a positive constant to achieve a number in between the flex range.
#     """
#     template_one_placeholder(np.arange(1, int(MAXIMUM_FLEX_VALUE)), x, ng.sqrt(x), lambda y: np.sqrt(y))