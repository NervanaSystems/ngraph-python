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
import ngraph as ng
from ngraph.testing import executor, assert_allclose


def get_executor_result(arg_array1, arg_array2, ng_placeholder1, ng_placeholder2, ng_fun):
    with executor(ng_fun, ng_placeholder1, ng_placeholder2) as m_executor:
        result = m_executor(arg_array1, arg_array2)
        return result


def template_create_placeholder_and_variable(n, c, const_val=0.1):
    N = ng.make_axis(length=n,  name="N")
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


def template_one_placeholder(values, ng_fun, ng_placeholder, expected_values, description):
    with executor(ng_fun, ng_placeholder) as const_executor:
        print(description)
        for value, expected_value in zip(values, expected_values):
            flex = const_executor(value)
            print("flex_value: ", flex)
            print("expected_value: ", expected_value)
            print(flex - expected_value)
            assert flex == expected_value


def template_two_placeholders(tuple_values, ng_fun, ng_placeholder1, ng_placeholder2,
                              expected_values, description):
    with executor(ng_fun, ng_placeholder1, ng_placeholder2) as const_executor:
        print(description)
        for values, expected_value in zip(tuple_values, expected_values):
            flex = const_executor(values[0], values[1])
            print("flex_value: ", flex)
            print("expected_value: ", expected_value)
            print(flex - expected_value)
            assert flex == expected_value


def template_dot_one_placeholder(n, c, const_val, expected_result):
    ng_placeholder, ng_variable = template_create_placeholder_and_variable(n, c, const_val)
    ng_fun = ng.dot(ng_variable, ng_placeholder)
    arg_array = np.array([i for i in range(c * n)]).reshape(c, n)
    ar_2 = np.copy(arg_array)
    expected_index = 0
    with executor(ng_fun, ng_placeholder) as mm_executor:
        for i in range(n):
            print (i)
            arg_array = ar_2 * (i + 1)
            print ("arg_array", arg_array)
            ng_op_out = mm_executor(arg_array)
            np_op_out = np.dot(np.ones(c) * const_val, arg_array)
            print ("ng_op_out", ng_op_out)
            print ("np_op_out", np_op_out)
            try:
                assert_allclose(ng_op_out, np_op_out)
            except AssertionError:
                print (list(ng_op_out))
                print (expected_result[expected_index])
                assert list(ng_op_out) == expected_result[expected_index]
                expected_index += 1


def template_dot_two_placeholders(n, c, d):
    ng_placeholder2, ng_placeholder1 = template_create_placeholders_for_multiplication(n, c, d)
    ng_fun = ng.dot(ng_placeholder1, ng_placeholder2)
    arg_array1 = np.array([i for i in range(c * d)]).reshape(d, c)
    arg_array2 = np.array([i for i in range(c * n)]).reshape(c, n)
    print ("arg_array1", arg_array1)
    print ("arg_array2", arg_array2)
    with executor(ng_fun, ng_placeholder1, ng_placeholder2) as mm_executor:
        np_op_out = np.dot(arg_array1, arg_array2)
        ng_op_out = mm_executor(arg_array1, arg_array2)
        print ("np_op_out\n", np_op_out)
        print ("ng_op_out\n", ng_op_out)
        assert_allclose(ng_op_out, np_op_out)


def template_dot_one_placeholder_and_scalar(n, c, scalar, flex_exceptions):
    arg_array = np.array([i for i in range(c * n)]).reshape(n, c)
    ng_placeholder = template_create_placeholder(n, c)
    ng_var = ng.placeholder(())
    ng_fun = ng.dot(ng_var, ng_placeholder)
    flex_exceptions_index = 0
    print ("first scalar:", scalar)
    print ("arg_array\n", arg_array)
    with executor(ng_fun, ng_var, ng_placeholder) as m_executor:
        for i in range (n):
            ng_op_out = m_executor(scalar, arg_array)
            np_op_out = np.dot(scalar, arg_array)
            arg_array = ng_op_out
            print("ng_op_out\n", ng_op_out)
            print("np_op_out\n", np_op_out)
            try:
                assert_allclose(ng_op_out, np_op_out)
            except AssertionError:
                print ((ng_op_out))
                print (flex_exceptions[flex_exceptions_index])
                assert np.equal(ng_op_out, flex_exceptions[flex_exceptions_index])



