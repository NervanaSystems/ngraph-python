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
from ngraph.op_graph.comm_nodes import calculate_new_axes
import ngraph as ng
import pytest


ax_A = ng.make_axis(length=10, name='A')
ax_B = ng.make_axis(length=15, name='B')
ax_C = ng.make_axis(length=20, name='C')
axes = ng.make_axes([ax_A, ax_B, ax_C])


@pytest.mark.parametrize("is_last", [False, True])
def test_calculate_new_axes_single_device(is_last):
    new_axes = calculate_new_axes(axes=axes, parallel_axis=ax_B, num_devices=1, is_last=is_last)
    assert new_axes.full_lengths == axes.full_lengths


@pytest.mark.parametrize("is_last, expected_B_length", [(False, 7), (True, 8)])
def tests_calculate_new_axes_two_devices(is_last, expected_B_length):
    new_axes = calculate_new_axes(axes=axes, parallel_axis=ax_B, num_devices=2, is_last=is_last)
    expected_axes = ng.make_axes([ax_A, ng.make_axis(length=expected_B_length, name='B'), ax_C])
    assert new_axes.full_lengths == expected_axes.full_lengths


def test_calculate_new_axes_zero_devices():
    with pytest.raises(ZeroDivisionError):
        calculate_new_axes(axes=axes, parallel_axis=ax_B, num_devices=0, is_last=False)


def test_calculate_new_axes_null_axes():
    with pytest.raises(TypeError):
        calculate_new_axes(axes=None, parallel_axis=ax_B, num_devices=2, is_last=True)


def test_calculate_new_axes_null_parallel_axis():
    new_axes = calculate_new_axes(axes=axes, parallel_axis=None, num_devices=1, is_last=False)
    # Checks null parallel axis. The axes calculated should have the same length as original
    assert new_axes.full_lengths == axes.full_lengths
