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
from __future__ import division
from operator import itemgetter

from ngraph.op_graph import op_graph
from ngraph.op_graph.axes import make_axis, Axes, spatial_axis


class pooling(op_graph.TensorOp):
    _index = 0

    def __init__(self, pool_params, inputs, *args, **kwargs):
        """
        Arguments:
            inputs  : input tensor.

        Return:
        """
        if len(inputs.shape) != 5:
            raise ValueError((
                'pooling input shape must be length 5, found {}'
            ).format(len(inputs.shape)))

        if 'axes' in kwargs:
            raise ValueError(
                "pooling does not currently support the 'axes' argument.  The "
                "output axes are entirely determined by the shape of the "
                "input and pooling params"
            )

        batch_axes = inputs.axes.batch_axes()
        if len(batch_axes) != 1:
            raise ValueError((
                "Input must have one batch axis.  Found {n_batch_axes} batch "
                "axes: {batch_axes} and {n_sample_axes} sample axes: "
                "{sample_axes}."
            ).format(
                n_batch_axes=len(batch_axes),
                batch_axes=batch_axes,
                n_sample_axes=len(inputs.axes.sample_axes()),
                sample_axes=inputs.axes.sample_axes(),
            ))
        pooltype = pool_params['op']
        if pooltype not in ('max', 'avg'):
            raise ValueError((
                "Unsupported pooling type: {pooltype}.  Only max and avg pooling "
                "currently supported. ").format(pooltype=pooltype))

        self.batch_axis = batch_axes[0]
        J, T, R, S = itemgetter(*('J', 'T', 'R', 'S'))(pool_params)
        axes = Axes(
            [spatial_axis(inputs, J, pool_params['pad_c'], pool_params['str_c'], role='channel'),
             spatial_axis(inputs, T, pool_params['pad_d'], pool_params['str_d'], role='depth'),
             spatial_axis(inputs, R, pool_params['pad_h'], pool_params['str_h'], role='height'),
             spatial_axis(inputs, S, pool_params['pad_w'], pool_params['str_w'], role='width'),
             self.batch_axis])

        self.pool_params = pool_params
        self.index = pooling._index

        pooling._index += 1

        super(pooling, self).__init__(
            args=(inputs,), *args, axes=axes, **kwargs
        )

    def generate_adjoints(self, adjoints, delta, inputs):
        inputs.generate_add_delta(adjoints, bprop_pool(delta, inputs, self))


class bprop_pool(op_graph.TensorOp):
    def __init__(self, delta, inputs, fprop, *args, **kwargs):
        """
        Arguments:
            inputs  : input tensor.
        """
        self.pool_params = fprop.pool_params
        self.index = fprop.index

        super(bprop_pool, self).__init__(
            args=(delta,), *args, axes=inputs.axes, **kwargs
        )
