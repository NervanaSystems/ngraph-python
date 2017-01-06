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

import ngraph.transformers as transformers
from ngraph.op_graph.axes import make_axis_role, make_axis, make_axes

from ngraph.op_graph.convolution import convolution
from ngraph.op_graph.pooling import pooling
from ngraph.op_graph.debug import PrintOp
from ngraph.op_graph.op_graph import *
from ngraph.op_graph.op_graph import axes_with_order, axes_with_role_order, \
    broadcast, cast_axes, \
    is_constant, is_constant_scalar, constant_value, constant_storage, \
    persistent_tensor, placeholder, init_tensor, \
    slice_along_axis, temporary, \
    add, as_op, as_ops, constant, variable, persistent_tensor, placeholder, \
    temporary, constant_value, variance, squared_L2, \
    negative, absolute, sin, cos, tanh, exp, log, reciprocal, safelog, sign, \
    square, sqrt, tensor_size, assign, batch_size, pad, sigmoid, \
    one_hot, stack
from ngraph.util.names import name_scope, with_name_scope, make_name_scope
import ngraph.testing as testing

__all__ = [
    'absolute',
    'add',
    'as_op',
    'as_ops',
    'axes_with_order',
    'batch_size',
    'broadcast',
    'cast_axes',
    'computation',
    'constant',
    'constant_value',
    'convolution',
    'cos',
    'exp',
    'is_constant',
    'is_constant_scalar',
    'log',
    'make_axes',
    'make_axis',
    'make_axis_role',
    'make_name_scope',
    'name_scope',
    'negative',
    'one_hot',
    'pad',
    'persistent_tensor',
    'placeholder',
    'pooling',
    'reciprocal',
    'safelog',
    'sequential',
    'sequential_op_factory',
    'sigmoid',
    'sign',
    'sin',
    'slice_along_axis',
    'sqrt',
    'square',
    'squared_L2',
    'stack',
    'tanh',
    'temporary',
    'testing',
    'tensor_size',
    'tensor_slice',
    'variable',
    'variance',
    'with_name_scope',
]


try:
    from ngraph.transformers.gputransform import GPUTransformer
except ImportError:
    pass
