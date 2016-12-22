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
import pytest
from ngraph.frontends.caffe2.c2_importer.utils import shape_to_axes, args_shape_to_axes
from ngraph.op_graph.axes import Axis


def expected_shape_to_axes(axes, expected):
    assert isinstance(axes, list)
    assert len(axes) == len(expected)
    ax1, ax2 = axes[0], axes[1]
    assert isinstance(ax1, Axis)
    assert isinstance(ax2, Axis)
    assert ax1.length == expected[0]
    assert ax2.length == expected[1]


def test_shape_to_axes():
    shape = [2, 3]
    axes = shape_to_axes(shape)
    expected_shape_to_axes(axes, shape)


def test_args_shape_to_axes():
    const_val = 5.
    shape = [2, 3]
    name_val = 'dummy'

    @args_shape_to_axes(1)
    def funct(const, axes, name):
        assert const == const_val
        expected_shape_to_axes(axes, shape)
        assert name == name_val
    funct(const_val, shape, name_val)


@pytest.mark.xfail(strict=True)
def test_args_shape_to_axes_wrong_pos():
    const_val = 5.
    shape = [2, 3]
    name_val = 'dummy'

    @args_shape_to_axes(2)
    def funct(const, axes, name):
        assert const == const_val
        expected_shape_to_axes(axes, shape)
        assert name == name_val
    funct(const_val, shape, name_val)
