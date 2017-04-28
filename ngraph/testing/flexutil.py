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
from types import FunctionType
import numpy as np
import ngraph as ng
from ngraph.testing import executor, assert_allclose

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


def get_placeholder_from_operand(operand):
    if not isinstance(operand, np.ndarray):
        return ng.placeholder(())
    else:
        return ng.placeholder(ng.make_axes([ng.make_axis(length=operand.size)]))


def id_func(param):
    description = ""
    if isinstance(param, str):
        return param
    elif isinstance(param, FunctionType):
        return param.func_name.title()
    elif isinstance(param, list):
        if len(param) != 1:
            description = " of iterating ("
        for i in param:
            description += " of {} equals {} | ".format(i[0], i[1])
            if len(i) > 2:
                description += ", because of {}".format(i[2])
        if len(param) != 1:
            return description + ")"

    return " "


def template_one_placeholder(operands, ng_fun):
    print("first operand: ", operands[0][0])
    print("second operand: ", operands[0][1])
    if not isinstance(operands[0][0], np.ndarray):
        ng_placeholder = ng.placeholder(())
    else:
        ng_placeholder = ng.placeholder(ng.make_axes([ng.make_axis(length=operands[0][0].size)]))

    with executor(ng_fun(ng_placeholder), ng_placeholder) as const_executor:
        print("operants: ", operands)
        print("ng_fun: ", ng_fun)
        print("ng_placeholder: ", ng_placeholder)
        iterations = len(operands) != 1
        for i in operands:
            print(i)
            print("Operand: ", i[0])
            print("Expected result: ", i[1])
            flex_result = const_executor(i[0])
            if len(i) > 2:
                print("Description: ", i[2])
            print("flex_result: ", flex_result)
            if iterations:
                assert_allclose(flex_result, i[1])
            elif not isinstance(operands[0][0], np.ndarray):
                assert(flex_result == i[1])
            else:
                assert np.array_equal(flex_result, i[1])


def template_two_placeholders(operands, ng_fun):
    print("first operand: ", operands[0][0])
    print("second operand: ", operands[0][1])
    ng_placeholder1 = get_placeholder_from_operand(operands[0][0])
    ng_placeholder2 = get_placeholder_from_operand(operands[0][1])
    iterations = len(operands) != 1
    with executor(ng_fun(ng_placeholder1, ng_placeholder2),
                  ng_placeholder1, ng_placeholder2) as const_executor:
        for i in operands:
            print("Operand 1: ", i[0])
            print("Operand 2: ", i[1])
            print("Expected result: ", i[2])
            flex_result = const_executor(i[0], i[1])
            temporary_numpy = np.maximum(i[0], i[1])
            print("flex_result: ", flex_result)
            print("Temporary NumPy results: ", temporary_numpy)
            print("difference: ", flex_result - i[2])
            if iterations:
                assert_allclose(flex_result, i[2])
            elif not isinstance(operands[0][0], np.ndarray):
                assert flex_result == i[2]
            else:
                assert_allclose(flex_result, i[2])


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
