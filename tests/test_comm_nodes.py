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


def test_calculate_new_axes_single_device():
    new_axes = calculate_new_axes(axes=axes, parallel_axis=ax_B, num_devices=1)
    assert new_axes.full_lengths == axes.full_lengths


@pytest.mark.parametrize("axis, num", [(ax_A, 2), (ax_B, 3), (ax_C, 4), (ax_A, 5), (ax_B, 5),
                                       (ax_C, 5)])
def test_calculate_new_axes_no_reminder(axis, num):
    new_axes = calculate_new_axes(axes=axes, parallel_axis=axis, num_devices=num)
    expected_axes = ng.make_axes(
        [a if a != axis else ng.make_axis(length=axis.length / num, name=a.name) for a in axes])
    assert new_axes.full_lengths == expected_axes.full_lengths


@pytest.mark.parametrize("axis, num", [(ax_B, 2), (ax_A, 3), (ax_B, 4), (ax_B, 6), (ax_C, 7)])
def tests_calculate_new_axes_has_reminder(axis, num):
    with pytest.raises(AssertionError):
        calculate_new_axes(axes=axes, parallel_axis=axis, num_devices=num)


def test_calculate_new_axes_zero_devices():
    with pytest.raises(ZeroDivisionError):
        calculate_new_axes(axes=axes, parallel_axis=ax_B, num_devices=0)


def test_calculate_new_axes_null_axes():
    with pytest.raises(TypeError):
        calculate_new_axes(axes=None, parallel_axis=ax_B, num_devices=2)


def test_calculate_new_axes_null_parallel_axis():
    new_axes = calculate_new_axes(axes=axes, parallel_axis=None, num_devices=1)
    # Checks null parallel axis. The axes calculated should have the same length as original
    assert new_axes.full_lengths == axes.full_lengths
