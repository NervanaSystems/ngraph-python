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
from ngraph.testing.flexutil import template_one_placeholder, template_two_placeholders, MINIMUM_FLEX_VALUE, \
    MAXIMUM_FLEX_VALUE

x = ng.placeholder(())
z = ng.placeholder(())

bug_1424 = pytest.mark.xfail(strict=True, reason="GitHub issue #1424, for ng.argmax and ng.argmin, "
                                                 "the values outside of the flex range are computed")

test_data_double_operand = (
    # template:(operation, operand, expected_result, description)

    # test_sum
    (ng.sum, [np.array([MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE / 2])], [MINIMUM_FLEX_VALUE],
     "Redop sum function - values from flex range, result expected to underflow"),
    (ng.sum, [np.array([MAXIMUM_FLEX_VALUE, 2.0])], [MAXIMUM_FLEX_VALUE],
     "Redop sum function - values from flex range, result expected to overflow"),
    (ng.sum, [np.array([MAXIMUM_FLEX_VALUE, -1.0, -2.0])], [MAXIMUM_FLEX_VALUE - 1.0 - 2.0],
     "Redop sum function - values from flex range, near positive boundary"),
    (ng.sum, [np.array([1, 2, 14, 4, 5, 6, 7, 8, 9, -1])], [55.0], "Redop sum function - values from flex range"),
    # For iterations assert_allclose is used so expected results should not have been very precise
    (ng.sum, [np.array([1.0, 2.0, 3.0, 4.0],),
              np.array([0.4, 0.03, 0.44, 1.47]),
              np.array([100, 2000, 3500.4, 10000])], [10.0, 2.33984375, 31.9990234375],
     "Redop sum function with iterations - values from flex range"),

    # test_prod
    (ng.prod, [np.array([MINIMUM_FLEX_VALUE, 2.0])], [MINIMUM_FLEX_VALUE],
     "Redop product function - values from flex range, result expected to underflow"),
    (ng.prod, [np.array([MAXIMUM_FLEX_VALUE, 2.0])], [MAXIMUM_FLEX_VALUE],
     "Redop product function - values from flex range, result expected to overflow"),
    (ng.prod, [np.array([MINIMUM_FLEX_VALUE / 10.0, 1.0, 2.0])], [-6553.5],
     "Redop product function - values from flex range"),
    (ng.prod, [np.array([MAXIMUM_FLEX_VALUE / 10.0, 1.0, 1.0])], [3276.625],
     "Redop product function - value from flex range, multiplied by 1.0"),
    (ng.prod, [np.array([MAXIMUM_FLEX_VALUE, 0.0, 5.0])], [0],
     "Redop product function - values from flex range, multiplied by zero"),
    (ng.prod, [np.array([1.0, 2.0, 3.0, 4.0])], [24.0],
     "Redop product function - values from flex range"),
    # For iterations assert_allclose is used so expected results should not have been very precise
    (ng.prod, [np.array([1.0, 2.0, 3.0, 4.0]),
               np.array([100, 200, 3, 4]),
               np.array([0.4, 100, 0.7, 10000])], [24.0, 31.9990234375, 127.99609375],
     "Redop product function with iterations - values from flex range"),

    # test_max
    (ng.max, [np.array([MAXIMUM_FLEX_VALUE - 2.0, MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE + 2.0])],
     [MAXIMUM_FLEX_VALUE], "Redop max function - result expected to overflow"),
    (ng.max, [np.array([MAXIMUM_FLEX_VALUE + 2.0, MAXIMUM_FLEX_VALUE + 2.0, MAXIMUM_FLEX_VALUE + 2.0])],
     [MAXIMUM_FLEX_VALUE], "Redop max function - result expected to overflow"),
    (ng.max, [np.array([MAXIMUM_FLEX_VALUE - 2.0, MAXIMUM_FLEX_VALUE])], [MAXIMUM_FLEX_VALUE],
     "Redop max function - values from flex range"),
    # For iterations assert_allclose is used so expected results should not have been very precise
    (ng.max, [np.array([0.4, 0.1, 0.2]),
              np.array([0, 100, 10]),
              np.array([1, 7, 0])], [0.3999, 0.4999, 1.9999], "Iterations max(x)"),

    # test_min
    (ng.min, [np.array([MINIMUM_FLEX_VALUE + 2.0, MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE - 2])],
     [MINIMUM_FLEX_VALUE], "Redop min function - result expected to underflow"),
    (ng.min, [np.array([MINIMUM_FLEX_VALUE + 2.0, MINIMUM_FLEX_VALUE])], [MINIMUM_FLEX_VALUE],
     "Redop min function - values from flex range"),
    # For iterations assert_allclose is used so expected results should not have been very precise
    (ng.min, [np.array([MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE + 1, MINIMUM_FLEX_VALUE + 2]),
              np.array([0.4, 0.39, 0.38]),
              np.array([MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE - 1, MAXIMUM_FLEX_VALUE - 2])],
     [MINIMUM_FLEX_VALUE, 0, MAXIMUM_FLEX_VALUE - 2], "Iterations min(x)"),

    # test_argmax
    bug_1424((ng.argmax, [np.array([MAXIMUM_FLEX_VALUE - 2.0, MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE + 2.0])], [1],
              "Redop argmax function - result expected to overflow")),
    (ng.argmax, [np.array([MAXIMUM_FLEX_VALUE - 2.0, MAXIMUM_FLEX_VALUE])], [1],
     "Redop argmax function - values from flex range"),
    (ng.argmax, [np.array([1, 0, 1])], [0],
     "Redop argmax function - values from flex range"),

    # test_argmin
    bug_1424((ng.argmin, [np.array([MINIMUM_FLEX_VALUE + 2.0, MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE - 2])], [1],
              "Redop argmin function - result expected to underflow")),
    (ng.argmin, [np.array([MINIMUM_FLEX_VALUE + 2.0, MINIMUM_FLEX_VALUE])], [1],
     "Redop argmin function - values from flex range")
 )

test_data_single_operand = (
    # template:(operation, operand_tuple, expected_result, description)

    # test_maximum
    (ng.maximum, [[MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE - 1, MAXIMUM_FLEX_VALUE - 2],
                  [MAXIMUM_FLEX_VALUE - 1, MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE - 1]],
     [MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE - 1],
     "Maximum function - result expected to overflow"),
    (ng.maximum, [(MAXIMUM_FLEX_VALUE - 2.0, MAXIMUM_FLEX_VALUE)], [MAXIMUM_FLEX_VALUE],
     "Maximum function - values from flex range"),

    # test_minimum
    (ng.minimum, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE - 2)], [MINIMUM_FLEX_VALUE],
     "Minimum function - result expected to underflow"),
    (ng.minimum, [(MINIMUM_FLEX_VALUE + 2.0, MINIMUM_FLEX_VALUE)], [MINIMUM_FLEX_VALUE],
     "Minimum function - values from flex range"),
)


@pytest.mark.parametrize("operation, operand, expected_result, description", test_data_double_operand)
def test_double_operand(transformer_factory, operation, operand, expected_result, description):
    y = ng.placeholder(ng.make_axis(length=len(operand[0])))
    template_one_placeholder(operand, operation(y), y, expected_result, description)


@pytest.mark.parametrize("operation, operand_tuple, expected_result, description", test_data_single_operand)
def test_single_operand(transformer_factory, operation, operand_tuple, expected_result, description):
    template_two_placeholders(operand_tuple, operation(x, z), x, z, expected_result, description)


test_data_single_operand2 = (
    # template:(operation, operand_tuple, expected_result, description)

    # test_maximum
    (ng.maximum, [[MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE - 1, MAXIMUM_FLEX_VALUE - 2],
                  [MAXIMUM_FLEX_VALUE - 1, MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE - 1]],
     [MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE, MAXIMUM_FLEX_VALUE - 1],
     "Maximum function - result expected to overflow"),
    (ng.maximum, [(MAXIMUM_FLEX_VALUE - 2.0, MAXIMUM_FLEX_VALUE)], [MAXIMUM_FLEX_VALUE],
     "Maximum function - values from flex range"),

    # test_minimum
    (ng.minimum, [(MINIMUM_FLEX_VALUE, MINIMUM_FLEX_VALUE - 2)], [MINIMUM_FLEX_VALUE],
     "Minimum function - result expected to underflow"),
    (ng.minimum, [(MINIMUM_FLEX_VALUE + 2.0, MINIMUM_FLEX_VALUE)], [MINIMUM_FLEX_VALUE],
     "Minimum function - values from flex range"),
)
