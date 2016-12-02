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
from ngraph.op_graph.op_graph import TensorOp, ContiguousOp


def convolution(conv_params, inputs, filters, axes, docstring=None):
    """

    Args:
        conv_params: Dimensions.
        inputs (TensorOp): The input tensor.
        filters (TensorOp): Filter/kernel tensor.
        docstring (String, optional): Documentation for the op.

    Returns:
        TensorOp: The result of the convolution.

    """
    return ConvolutionOp(conv_params, inputs, filters, axes=axes, docstring=docstring)


class ConvolutionOp(TensorOp):
    _index = 0

    def __init__(self, conv_params, inputs, filters, **kwargs):
        """
        Arguments:
            inputs  : input tensor.
            filters : filter/kernel tensor.

        Return:
        """
        if len(inputs.shape) != 5:
            raise ValueError((
                'convolution input shape must be length 5, found {}'
            ).format(len(inputs.shape)))

        if len(filters.shape) != 5:
            raise ValueError((
                'convolution filter shape must be length 5, found {}'
            ).format(len(filters.shape)))

        if inputs.axes[0] != filters.axes[0]:
            raise ValueError((
                'the first axis in input {inputs} and filter {filters} are not the same.'
            ).format(inputs=inputs.axes[0], filters=filters.axes[0]))

        batch_axes = inputs.axes.batch_axes()
        if len(batch_axes) != 1:
            raise ValueError((
                "Input must have one batch axis.  "
                "Found {n_batch_axes} batch axes: {batch_axes} "
                "Found {n_sample_axes} sample axes: {sample_axes}."
            ).format(
                n_batch_axes=len(batch_axes),
                batch_axes=batch_axes,
                n_sample_axes=len(inputs.axes.sample_axes()),
                sample_axes=inputs.axes.sample_axes(),
            ))

        self.conv_params = conv_params
        self.index = ConvolutionOp._index
        ConvolutionOp._index += 1

        super(ConvolutionOp, self).__init__(
            args=(ContiguousOp(inputs), ContiguousOp(filters)), **kwargs
        )

    def generate_adjoints(self, adjoints, delta, inputs, filters):
        """
        TODO
        """
        filters.generate_add_delta(adjoints, update_conv(delta, inputs, filters, self))
        inputs.generate_add_delta(adjoints, bprop_conv(delta, inputs, filters, self))


class update_conv(TensorOp):
    def __init__(self, delta, inputs, filters, fprop, **kwargs):
        """
        Arguments:
            inputs  : input tensor.
            filters : filter/kernel tensor.
        """
        self.conv_params = fprop.conv_params
        self.index = fprop.index

        super(update_conv, self).__init__(
            args=(ContiguousOp(delta), ContiguousOp(inputs)),
            axes=filters.axes, **kwargs
        )


class bprop_conv(TensorOp):
    def __init__(self, delta, inputs, filters, fprop, **kwargs):
        """
        Arguments:
            inputs  : input tensor.
            filters : filter/kernel tensor.
        """
        self.fprop = fprop

        self.conv_params = fprop.conv_params
        self.index = fprop.index

        super(bprop_conv, self).__init__(
            args=(ContiguousOp(delta), ContiguousOp(filters)),
            axes=inputs.axes, **kwargs
        )
