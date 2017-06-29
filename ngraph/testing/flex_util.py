# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
from types import FunctionType
import numpy as np
import ngraph as ng
from ngraph.testing import executor, assert_allclose, RandomTensorGenerator, reference_conv, \
    ConvParams

MINIMUM_FLEX_VALUE = -2 ** 15
MAXIMUM_FLEX_VALUE = 2 ** 15 - 1


def get_executor_result(arg_array1, arg_array2, ng_placeholder1, ng_placeholder2, ng_fun):
    with executor(ng_fun, ng_placeholder1, ng_placeholder2) as m_executor:
        result = m_executor(arg_array1, arg_array2)
        return result


def template_create_placeholder_and_variable(n, c, const_val=0.1):
    N = ng.make_axis(length=n, name="N")
    C = ng.make_axis(length=c, name="C")
    X = ng.placeholder(axes=(C, N))
    W = ng.variable(axes=[C], initial_value=const_val)
    return X, W


def template_create_placeholder(n, c):
    N = ng.make_axis(length=n, name="N")
    C = ng.make_axis(length=c, name="C")
    return ng.placeholder((N, C))


def template_create_placeholders_for_multiplication(n, c, d):
    N = ng.make_axis(length=n, name="N")
    C = ng.make_axis(length=c, name="C")
    D = ng.make_axis(length=d, name="D")

    X = ng.placeholder((C, N))
    Y = ng.placeholder((D, C))
    return X, Y


def get_placeholder_from_operand(operand, axes=None):
    if not isinstance(axes, ng.Axes):
        if not isinstance(operand, np.ndarray):
            axes = ()
        else:
            if len(operand.shape) > 1:
                rows, columns = operand.shape
                N = ng.make_axis(length=rows)
                M = ng.make_axis(length=columns)
                axes = ng.make_axes([N, M])
            else:
                O = ng.make_axis(length=operand.size)
                axes = ng.make_axes([O])
    return ng.placeholder(axes), axes


def id_func(param):
    description = ""
    if isinstance(param, str):
        return param
    elif isinstance(param, FunctionType):
        return param.func_name.title()
    elif isinstance(param, list):
        if len(param) > 1:
            description += "Iterations: ("
        for i in param:
            operands, result, explanation = unpack_list(*i)
            if len(operands) == 1:
                description += " of {}".format(operands[0])
            else:
                description += " {} of {}".format(operands[0], operands[1])
            description += " equals {} ".format(result)
            if explanation:
                description += ", because of {} ".format(explanation)
            description += "|"
        if len(param) > 1:
            description += ")"
        return description
    elif isinstance(param, tuple):
        for i in param:
            description += str(i)
        return description
    return ""


def unpack_list(a, b, *c):
    if c:
        if len(c) > 1:
            return (a, b), c[0], c[1:]
        elif isinstance(c[0], str):
            return (a,), b, c
        else:
            return (a, b), c[0], None
    else:
        return (a,), b, None


def execute_calculation(operands, first_operand, const_executor):
    iterations = len(operands) != 1
    for i in operands:
        _operands, expected_result, description = unpack_list(*i)
        if description:
            print("Description: ", description)
        print("Operands: ", _operands)
        print("Expected result: ", expected_result)
        flex_result = const_executor(*_operands)
        try:
            print("flex_result: {0:.30}".format(float(flex_result)))
        except TypeError:
            # exception for arrays
            np.set_printoptions(precision=30)
            print("flex_result: {}".format(flex_result))
        print("difference: ", flex_result - expected_result)
        if iterations:
            assert_allclose(flex_result, expected_result)
        elif not isinstance(first_operand, np.ndarray):
            assert flex_result == expected_result
        else:
            assert np.array_equal(flex_result, expected_result)


def template_one_placeholder(operands, ng_fun):
    first_operand = operands[0][0]
    ng_placeholder, _ = get_placeholder_from_operand(first_operand)

    with executor(ng_fun(ng_placeholder), ng_placeholder) as const_executor:
        execute_calculation(operands, first_operand, const_executor)


def template_two_placeholders(operands, ng_fun):
    first_operand = operands[0][0]
    second_operand = operands[0][1]

    ng_placeholder1, axes = get_placeholder_from_operand(first_operand)
    ng_placeholder2, _ = get_placeholder_from_operand(second_operand, axes=axes)

    with executor(ng_fun(ng_placeholder1, ng_placeholder2),
                  ng_placeholder1, ng_placeholder2) as const_executor:
        execute_calculation(operands, first_operand, const_executor)


def template_dot_one_placeholder(row, col, const_val, flex_exceptions, iters):
    ng_placeholder, ng_variable = template_create_placeholder_and_variable(col, row, const_val)
    ng_fun = ng.dot(ng_variable, ng_placeholder)
    arg_array = np.array([i for i in range(row * col)]).reshape(row, col)
    arg_array2 = np.copy(arg_array)
    flex_exceptions_index = 0
    vector = np.ones(row) * const_val
    print("Vector: \n", vector)
    with executor(ng_fun, ng_placeholder) as mm_executor:
        for i in range(iters):
            print("Iteration " + str(i + 1))

            # After each iteration, matrix values are changed
            arg_array = arg_array2 * (i + 1)

            print("Matrix: \n", arg_array)
            ng_op_out = mm_executor(arg_array)
            np_op_out = np.dot(vector, arg_array)
            print("Flex dot product result: \n ", ng_op_out)
            print("Numpy dot product result: \n", np_op_out)
            try:
                assert_allclose(ng_op_out, np_op_out)
            except AssertionError:
                print("Flex dot product result doesn't match to numpy.\n"
                      "Try to check if flex result is inside flex exceptions list")
                print("Flex dot product result: \n", ng_op_out)
                print("Current array inside flex exceptions list: \n",
                      flex_exceptions[flex_exceptions_index])

                assert list(ng_op_out) == flex_exceptions[flex_exceptions_index]

                # Iterate to the next element of flex exceptions list
                flex_exceptions_index += 1


def template_dot_two_placeholders(rows_1, col_1, col_2):
    ng_placeholder2, ng_placeholder1 = \
        template_create_placeholders_for_multiplication(col_2, col_1, rows_1)
    ng_fun = ng.dot(ng_placeholder1, ng_placeholder2)
    arg_array1 = np.array([i for i in range(col_1 * rows_1)]).reshape(rows_1, col_1)
    arg_array2 = np.array([i for i in range(col_1 * col_2)]).reshape(col_1, col_2)
    print("Matrix 1:\n", arg_array1)
    print("Matrix 2:\n", arg_array2)
    with executor(ng_fun, ng_placeholder1, ng_placeholder2) as mm_executor:
        np_op_out = np.dot(arg_array1, arg_array2)
        ng_op_out = mm_executor(arg_array1, arg_array2)
        print("Flex dot product result: \n", ng_op_out)
        print("Numpy dot product result: \n", np_op_out)
        assert_allclose(ng_op_out, np_op_out)


def template_dot_one_placeholder_and_scalar(row, col, scalar, flex_exceptions, iters):
    arg_array = np.array([i for i in range(col * row)]).reshape(row, col)
    ng_placeholder = template_create_placeholder(row, col)
    ng_var = ng.placeholder(())
    ng_fun = ng.dot(ng_var, ng_placeholder)
    flex_exceptions_index = 0
    print("Initial scalar: ", scalar)
    print("Matrix:\n", arg_array)
    with executor(ng_fun, ng_var, ng_placeholder) as m_executor:
        for i in range(row):
            print("Iteration " + str(i + 1))
            ng_op_out = m_executor(scalar, arg_array)
            np_op_out = np.dot(scalar, arg_array)

            # After each iteration matrix values are updated.
            arg_array = ng_op_out

            print("Flex dot product result: \n", ng_op_out)
            print("Numpy dot product result: \n", np_op_out)
            try:
                assert_allclose(ng_op_out, np_op_out)
            except AssertionError:
                print("Flex dot product result doesn't match to numpy.\n"
                      "Try to check if flex result is inside flex exceptions list")
                print("Flex dot product result: \n", ng_op_out)
                print("Current array inside flex exceptions list: \n",
                      flex_exceptions[flex_exceptions_index])
                assert_allclose(ng_op_out, flex_exceptions[flex_exceptions_index])

                # Iterate to the next element of flex exceptions list
                flex_exceptions_index += 1


def execute_convolution(image_height, image_width, filter_height, filter_width, channel=16,
                        batch_size=32, filter_count=8, image_3rd_dim=1, filter_3rd_dim=1,
                        padding=(0, 0, 0), stride=(1, 1, 1), dilation=1, np_comparison=False):

    pad_h, pad_w, pad_d = padding
    str_h, str_w, str_d = stride
    cf = ConvParams(C=channel, N=batch_size, K=filter_count, D=image_3rd_dim, H=image_height,
                    W=image_width, T=filter_3rd_dim, R=filter_height, S=filter_width,
                    pad_d=pad_d, pad_h=pad_h, pad_w=pad_w, str_d=str_d, str_h=str_h, str_w=str_w,
                    dil_d=dilation, dil_h=dilation, dil_w=dilation)

    inputs = ng.placeholder(cf.ax_i)
    filters = ng.placeholder(cf.ax_f)
    rng = RandomTensorGenerator(0, np.float32)
    input_value = rng.uniform(-4, 4, cf.ax_i, dtype=int)
    filter_value = rng.uniform(-4, 4, cf.ax_f, dtype=int)
    error_value = rng.uniform(-0.5, 0.5, cf.ax_o)
    with executor(ng.convolution(cf.conv_params, inputs, filters, axes=cf.ax_o),
                  inputs, filters) as const_executor:
        out = const_executor(input_value, filter_value)

    if np_comparison:
        np_out, gradInp, gradF_np = \
            reference_conv(cf.dimI, cf.dimF, cf.dimO, cf.conv_params, input_value, filter_value,
                           error_value)
        return out, np_out
    return out
