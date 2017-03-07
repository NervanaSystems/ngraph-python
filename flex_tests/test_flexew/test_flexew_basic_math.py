import numpy as np
import pytest

import ngraph as ng
from ngraph.testing import executor, template_one_placeholder, template_one_placeholder_equality




MINIMUM_FLEX_VALUE = -128  # 128.0
MAXIMUM_FLEX_VALUE = 127.99609375  # 127.99609375/
# =======
# MINIMUM_FLEX_VALUE = - 128.0
# MAXIMUM_FLEX_VALUE = 127.99609375
# >>>>>>> 34f5311cf66da29482b1c07217d9d06856bc182a
EPSILON = 0.2
x = ng.placeholder(())
z = ng.placeholder(())


# Assignment
# @pytest.mark.xfail(reason="Known issue")

testdata_assign = \
    [
     (MINIMUM_FLEX_VALUE - 1, MINIMUM_FLEX_VALUE, "Assign function - underflow expected"),
     (MAXIMUM_FLEX_VALUE + 1, MAXIMUM_FLEX_VALUE, "Assign function - overflow expected"),
     (MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE, "Assign function of negative boundary value inside of flex range"),
     (MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, "Assign function of positive boundary value inside of flex range")
    ]



@pytest.mark.parametrize("test_input, expected, description", testdata_assign)
def test_assign(transformer_factory, test_input, expected, description):
    # pytest.skip('Not for flex')
    template_one_placeholder(test_input, x, x, expected)

# else:
#     pytest.skip('Not for flex')
#
# def test_assign_const_below_minimum_flex_value(transformer_factory):
#     """
#     Assign constant with a value below the flex range.
#     Negative case.
#     """
#     template_one_placeholder([MAXIMUM_FLEX_VALUE +1], x, x, lambda y: y, expect_error=True)
#
#
# def test_assign_const_minimum_flex_value(transformer_factory):
#     """
#     Assign constant with the minimum value of the flex range.
#     """
#     template_one_placeholder([MINIMUM_FLEX_VALUE - 1], x, x, lambda y: y, expect_error=True)
#
#
# def test_assign_const_maximum_flex_value(transformer_factory):
#     """
#     Assign constant with the maximum value of the flex range.
#     """
#     template_one_placeholder([MINIMUM_FLEX_VALUE], x, x, lambda y: y)
#
#
#
# # @pytest.mark.xfail(reason="Known issue")
# def test_assign_const_above_maximum_flex_value(transformer_factory):
#     """
#     Assign constant with a value above the flex range.
#     Negative case.
#     """
#     template_one_placeholder([MAXIMUM_FLEX_VALUE + 1], x, x, lambda y: y, expect_error=True)
#
#
# # Negate.
# def test_neg_const_below_minimum_flex_value(transformer_factory):
#     """
#     Negate constant with a value below the flex range.
#     Negative case.
#     """
#     template_one_placeholder([MINIMUM_FLEX_VALUE - 1], x, -x, lambda y: -y, expect_error=True)
#
#
# def test_neg_const_minimum_flex_value(transformer_factory):
#     """
#     Negate constant with the minimum value from the flex range.
#     """
#     template_one_placeholder([MINIMUM_FLEX_VALUE], x, -x, lambda y: -y)
#
# def test_neg_const_maximum_flex_value(transformer_factory):
#     """
#     Negate constant with the maximum value from the flex range.
#     """
#     template_one_placeholder([MAXIMUM_FLEX_VALUE], x, -x, lambda y: -y)
#
#
# def test_neg_const_above_maximum_flex_value(transformer_factory):
#     """
#     Negate constant of a value above the flex range.
#     Negative case.
#     """
#     template_one_placeholder([MAXIMUM_FLEX_VALUE + 1], x, -x, lambda y: -y, expect_error=True)
#
#
# # Absolute value.
# def test_abs_matrix(transformer_factory):
#     """
#     absolute value of matrix
#     """
#     n, m = 2, 3
#     N = ng.make_axis(length=n)
#     M = ng.make_axis(length=m)
#     Zin = ng.placeholder((N, M))
#     Zout = abs(Zin)
#
#     with executor(Zout, Zin) as ex:
#         abs_executor = ex
#
#         Xval = np.array([5, 1, 0, -2, 3, 4]).reshape(n, m).astype(np.float32)
#         Xval[0, 1] = -Xval[0, 1]
#         assert np.allclose(abs_executor(Xval), abs(Xval))
#
#
# def test_abs_const_minimum_flex_value(transformer_factory):
#     """
#     Absolute value from the flex range.
#     """
#     template_one_placeholder(np.arange(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, 0.3), x, ng.absolute(x),
#                              lambda y: abs(y))
#
#
# def test_abs_const_below_flex_value(transformer_factory):
#     """
#     Absolute value with the value below the flex range.
#     Negative case
#     """
#     template_one_placeholder(np.arange(MINIMUM_FLEX_VALUE - 1, MINIMUM_FLEX_VALUE - 3, 0.3), x, ng.absolute(x),
#                              lambda y: abs(y), expect_error=True)
#
#
# def test_abs_const_above_flex_value(transformer_factory):
#     """
#     Absolute value with the value above the flex range.
#     Negative case
#     """
#     template_one_placeholder(np.arange(MAXIMUM_FLEX_VALUE + 1, MAXIMUM_FLEX_VALUE + 3, 0.3), x, ng.absolute(x),
#                              lambda y: abs(y), expect_error=True)
#
#
# # Addition.
# def test_addition_const_below_minimum_flex_value(transformer_factory):
#     """
#     Add constants to achieve a number below the flex range
#     Negative case
#     """
#     template_one_placeholder([MINIMUM_FLEX_VALUE - 2], x, x + 1.4, lambda y: y + 1.4, expect_error=True)
#
#
# def test_addition_const_from_flex_value(transformer_factory):
#     """
#     Add constants to achieve a number within the flex range
#     Number is the boundary value of minimum flex range
#     """
#     template_one_placeholder(np.arange(MINIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE - 1.4), x, x + 1.4, lambda y: y + 1.4)
#
#
# def test_addition_const_over_maximum_flex_value(transformer_factory):
#     """
#     Add constants to achieve a number over the flex range
#     Negative case
#     """
#     template_one_placeholder([MAXIMUM_FLEX_VALUE], x, x + 1.4, lambda y: y + 1.4, expect_error=True)
#
#
# def test_plusconst(transformer_factory):
#     """
#     x + 1.5
#     """
#     x = ng.placeholder(())
#     x_plus_const = x + 1.5
#
#     with executor(x_plus_const, x) as ex:
#         plusconst_executor = ex
#
#         for i in range(5):
#             # 8.8 fixed point test
#             assert plusconst_executor(i) == i + 1.5
#
#
# # Subtraction.
# def test_subtraction_const_below_minimum_flex_value(transformer_factory):
#     """
#     Subtract constants to achieve a number below the flex range
#     Negative case
#     """
#     template_one_placeholder([MINIMUM_FLEX_VALUE], x, x - 1, lambda y: y - 1, expect_error=True)
#
#
# def test_subtraction_const_minimum_flex_value(transformer_factory):
#     """
#     Subtract constants to achieve the minimum value of the flex range
#     """
#     template_one_placeholder([MINIMUM_FLEX_VALUE + 1], x, x - 1, lambda y: y - 1)
#
#
# def test_subtraction_const_maximum_flex_value(transformer_factory):
#     """
#     Subtract constants with maximum value of the flex range
#     """
#     template_one_placeholder([MAXIMUM_FLEX_VALUE], x, x - 1, lambda y: y - 1)
#
#
# def test_subtraction_const_above_maximum_flex_value(transformer_factory):
#     """
#     Subtract constants to achieve a number above the flex range
#     Negative case
#     """
#     template_one_placeholder([MAXIMUM_FLEX_VALUE + 2], x, x - 1, lambda y: y - 1, expect_error=True)
#
#
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