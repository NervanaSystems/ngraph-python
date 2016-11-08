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


def pooling(poolparams, inputs, axes, name=None, docstring=None):
    """

    Args:
        poolparams: Dimensions.
        inputs (TensorOp): Input to pooling.
        name (String, optional): Name of the Op.
        docstring (String, optional): Dcoumentation for the computation.

    Returns:
        TensorOp: The pooling computation.

    """
    return PoolingOp(poolparams, inputs, axes=axes, name=name, docstring=docstring)


class PoolingOp(op_graph.TensorOp):
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

        self.pool_params = pool_params
        self.index = PoolingOp._index

        PoolingOp._index += 1

        super(PoolingOp, self).__init__(
            args=(inputs,), *args, **kwargs
        )

    def generate_adjoints(self, adjoints, delta, inputs):
        inputs.generate_add_delta(adjoints, BpropPoolOp(delta, inputs, self))


class BpropPoolOp(op_graph.TensorOp):
    def __init__(self, delta, inputs, fprop, *args, **kwargs):
        """
        Arguments:
            inputs  : input tensor.
        """
        self.pool_params = fprop.pool_params
        self.index = fprop.index

        super(BpropPoolOp, self).__init__(
            args=(delta,), *args, axes=inputs.axes, **kwargs
        )
