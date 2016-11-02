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

from ngraph.op_graph import op_graph
from ngraph.op_graph.axes import Axis, Axes


class pooling(op_graph.TensorOp):
    _index = 0

    def __init__(self, dims, inputs, *args, **kwargs):
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
                "input and filter Ops."
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
        self.batch_axis = batch_axes[0]
        axes = Axes([Axis(dim) for dim in dims.dimO[:-1]]) + self.batch_axis
        for i, name in enumerate(['C', 'D', 'H', 'W']):
            axes[i].name = name

        self.dims = dims

        # TODO: argmax_axes expand
        self.argmax = ng.persistent_tensor(axes=argmax_axes, name='pool')
        self.index = pooling._index
        pooling._index += 1

        super(pooling, self).__init__(
            args=(inputs), *args, axes=axes, **kwargs
        )

    def generate_adjoints(self, adjoints, delta, inputs, argmax):
        inputs.generate_add_delta(adjoints, bprop_pool(delta, inputs, argmax, self))


class bprop_pool(op_graph.TensorOp):
    def __init__(self, delta, inputs, argmax, fprop, *args, **kwargs):
        """
        Arguments:
            inputs  : input tensor.
        """
        self.dims = fprop.dims
        self.index = fprop.index

        super(bprop_pool, self).__init__(
            args=(delta, argmax), *args, axes=inputs.axes, **kwargs
        )
