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
from ngraph.testing import executor


def template_one_placeholder(value, ng_fun, ng_placeholder, expected_value, description,
                             epsilon=0.2):
    with executor(ng_fun, ng_placeholder) as const_executor:
        print(description)
        flex = const_executor(value)
        print("flex_value: ", flex)
        print("expected_value: ", expected_value)
        print(flex - expected_value)
        assert (abs(flex - expected_value) <= epsilon)


def template_one_placeholder(value, ng_fun, ng_placeholder, expected_value, description, epsilon=0.2):
    with executor(ng_fun, ng_placeholder) as const_executor:
        print(description)
        if isinstance(value, list):
            for val, expected_val in zip(value, expected_value):
                flex = const_executor(val)
                print("flex_value: ", flex)
                print("expected_value: ", expected_val)
                print(flex - expected_val)
                assert (abs(flex - expected_val) <= epsilon)