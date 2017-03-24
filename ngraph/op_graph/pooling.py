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


def pooling(poolparams, inputs, axes, docstring=None):
    """

    Args:
        poolparams: Dimensions.
        inputs (TensorOp): Input to pooling.
        docstring (String, optional): Dcoumentation for the computation.

    Returns:
        TensorOp: The pooling computation.
    """
    return PoolingOp(poolparams, inputs, axes=axes, docstring=docstring)


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
        # requires pooling's forward to be completed before backward
        bprop_pool_op = BpropPoolOp(delta, inputs, self)
        bprop_pool_op.add_control_dep(self)
        inputs.generate_add_delta(adjoints, bprop_pool_op)


class PoolDerivOp(op_graph.TensorOp):
    """
    Maintains index and pool_params through forwarding of the original PoolingOp.

    Arguments:
        fprop: The original PoolingOp.
    """
    def __init__(self, fprop, **kwargs):
        super(PoolDerivOp, self).__init__(**kwargs)
        self.fprop = fprop

    @property
    def index(self):
        """

        Returns:
            The argmax tensor index of the pooling op.
        """
        return self.fprop.forwarded.index

    @property
    def pool_params(self):
        """

        Returns:
            The pooling parameters of the pooling op.

        """
        return self.fprop.forwarded.pool_params


class BpropPoolOp(PoolDerivOp):
    def __init__(self, delta, inputs, fprop, *args, **kwargs):
        """
        Arguments:
            inputs  : input tensor.
        """
        super(BpropPoolOp, self).__init__(
            args=(delta,),
            fprop=fprop,
            axes=inputs.axes,
            *args, **kwargs
        )

        self.inputs = inputs
