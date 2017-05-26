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
from ngraph.op_graph.op_graph import TensorValueOp
from ngraph.factory.comm_node_factory import get_comm_pattern
from ngraph.op_graph.comm_nodes import calculate_scatter_axes
from ngraph.frontends.neon import UniformInit
from contextlib import closing
import ngraph as ng
import ngraph.transformers as ngt
import pytest
import numpy as np


ax_A = ng.make_axis(length=10, name='A')
ax_B = ng.make_axis(length=15, name='B')
ax_C = ng.make_axis(length=20, name='C')
axes = ng.make_axes([ax_A, ax_B, ax_C])


def test_calculate_new_axes_single_device():
    new_axes = calculate_scatter_axes(axes=axes, scatter_axis=ax_B, num_devices=1)
    assert new_axes.full_lengths == axes.full_lengths


@pytest.mark.parametrize("axis, num", [(ax_A, 2), (ax_B, 3), (ax_C, 4), (ax_A, 5), (ax_B, 5),
                                       (ax_C, 5)])
def test_calculate_new_axes_no_remainder(axis, num):
    new_axes = calculate_scatter_axes(axes=axes, scatter_axis=axis, num_devices=num)
    expected_axes = ng.make_axes(
        [a if a != axis else ng.make_axis(length=axis.length / num, name=a.name) for a in axes])
    assert new_axes.full_lengths == expected_axes.full_lengths


@pytest.mark.parametrize("axis, num", [(ax_B, 2), (ax_A, 3), (ax_B, 4), (ax_B, 6), (ax_C, 7)])
def tests_calculate_new_axes_has_remainder(axis, num):
    with pytest.raises(AssertionError):
        calculate_scatter_axes(axes=axes, scatter_axis=axis, num_devices=num)


def test_calculate_new_axes_zero_devices():
    with pytest.raises(ZeroDivisionError):
        calculate_scatter_axes(axes=axes, scatter_axis=ax_B, num_devices=0)


def test_calculate_new_axes_null_axes():
    with pytest.raises(TypeError):
        calculate_scatter_axes(axes=None, scatter_axis=ax_B, num_devices=2)


def test_calculate_new_axes_null_parallel_axis():
    new_axes = calculate_scatter_axes(axes=axes, scatter_axis=None, num_devices=1)
    # Checks null parallel axis. The axes calculated should have the same length as original
    assert new_axes.full_lengths == axes.full_lengths


@pytest.mark.parametrize("from_node, to_node, expected_type", [
    (None, None, None),
    (
        ng.Op(metadata=dict(device='cpu', device_id='0', transformer='cpu0')),
        ng.Op(metadata=dict(device='cpu', device_id='0', transformer='cpu0')),
        None
    ),
    (
        ng.Op(metadata=dict(device='cpu', device_id='0', transformer='cpu0')),
        ng.Op(metadata=dict(device='cpu', device_id='1', transformer='cpu1')),
        'direct'
    ),
    (
        ng.Op(metadata=dict(device='cpu', device_id='0', transformer='cpu0')),
        ng.Op(metadata=dict(device='gpu', device_id='0', transformer='gpu0')),
        'direct'
    ),
    (
        TensorValueOp(ng.constant(1),
                      metadata=dict(device='cpu', device_id='0', transformer='cpu0')),
        ng.Op(metadata=dict(device='cpu', device_id=('1', '2'), parallel=ax_B,
              transformer=['cpu1', 'cpu2'])),
        None
    ),
    (
        TensorValueOp(ng.placeholder([ax_A, ax_B]),
                      metadata=dict(device='cpu', device_id='0', transformer='cpu0')),
        ng.Op(metadata=dict(device='cpu', device_id=('1', '2'), parallel=ax_B,
                            transformer=['cpu1', 'cpu2'])),
        'scatter'
    ),
    (
        TensorValueOp(ng.placeholder([ax_A, ax_B]),
                      metadata=dict(device='cpu', device_id='0', transformer='cpu0')),
        ng.Op(metadata=dict(device='cpu', device_id=('1', '2'), parallel=ax_C,
                            transformer=['cpu1', 'cpu2'])),
        'direct'
    ),
    (
        ng.Op(metadata=dict(device='cpu', device_id=('1', '2'), parallel=ax_C,
                            transformer=['cpu1', 'cpu2'])),
        ng.Op(metadata=dict(device='cpu', device_id='0', transformer='cpu0')),
        'gather'
    ),
    (
        ng.Op(metadata=dict(device='cpu', device_id=('1', '2'), parallel=ax_C,
                            transformer=['cpu1', 'cpu2'], reduce_func='mean')),
        ng.Op(metadata=dict(device='cpu', device_id=('1', '2'), parallel=ax_C,
                            transformer=['cpu1', 'cpu2'])),
        'allreduce'
    ),
])
def test_get_node_type(from_node, to_node, expected_type):
    assert expected_type == get_comm_pattern(from_node, to_node)


@pytest.mark.parametrize('config', [
    {
        'input': 36,
        'func': 'mean',
        'device_id': (1, 2),
        'expected_result': [[[-35.0, -35.0], [-35.0, -35.0], [-35.0, -35.0], [-35.0, -35.0]]],
    },
    {
        'input': 36,
        'func': 'sum',
        'device_id': (1, 2),
        'expected_result': [[[-71.0, -71.0], [-71.0, -71.0], [-71.0, -71.0], [-71.0, -71.0]]],
    },
    {
        'input': 36,
        'func': 'mean',
        'device_id': (1, 4, 3, 2),
        'expected_result': [[[-35.0, -35.0], [-35.0, -35.0], [-35.0, -35.0], [-35.0, -35.0]]],
    },
    {
        'input': 25,
        'func': 'sum',
        'device_id': (5, 7, 3, 4),
        'expected_result': [[[-99.0, -99.0], [-99.0, -99.0], [-99.0, -99.0], [-99.0, -99.0]]],
    },
])
def test_allreduce_hint(config):
    c = config

    with ng.metadata(device_id=c['device_id']):
        axis_A = ng.make_axis(length=4, name='axis_A')
        axis_B = ng.make_axis(length=2, name='axis_B')
        var_A = ng.variable(axes=[axis_A], initial_value=UniformInit(1, 1)).named('var_A')
        var_B = ng.variable(initial_value=UniformInit(c['input'], c['input']),
                            axes=[axis_B]).named('var_B')
        var_B.metadata['reduce_func'] = c['func']
        var_minus = (var_A - var_B).named('var_minus')
        var_minus.metadata['parallel'] = axis_A
    with closing(ngt.make_transformer_factory('hetr')()) as hetr:
        out_comp = hetr.computation([var_minus]).named('out_comp')
        result = out_comp()

        np.testing.assert_array_equal(result, c['expected_result'])


@pytest.mark.parametrize('config', [
    {
        'input': 1,
        'func': 'sum',
        'device_id': (1, 2),
        'expected_result': [-35.0, -35.0, -35.0, -35.0],
    },
])
def test_one_dot_bprop_allreduce(config):
    c = config

    pytest.xfail("GPU child transformers generate errors during AssignLayouts graph pass #1651")

    H_axis = ng.make_axis(length=4, name='height')
    W_axis = ng.make_axis(length=6, name='width')
    with ng.metadata(step='input'):
        X = ng.placeholder(axes=[H_axis, W_axis])
        target = ng.constant(1, axes=[W_axis])
    with ng.metadata(device_id=c['device_id'], parallel=W_axis):
        W = ng.variable(axes=[H_axis], initial_value=UniformInit(1, 1))
        dot = ng.dot(W, X)
        L = ng.squared_L2(target - dot, out_axes=())
        grad = ng.deriv(L, W)
        grad.metadata['reduce_func'] = c['func']
        update = (W - grad)

    with closing(ngt.make_transformer_factory('hetr')()) as hetr:
        out_comp = hetr.computation([update], X)
        result = out_comp(c['input'])

        np.testing.assert_array_equal(result, c['expected_result'])
