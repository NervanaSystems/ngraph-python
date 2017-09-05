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
from ngraph.op_graph.op_graph import TensorOp


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
    """
    Arguments:
        inputs  : input tensor.
        filters : filter/kernel tensor.

    Return:
    """

    def __init__(self, conv_params, inputs, filters, bias=None, **kwargs):
        if bias is None:
            super(ConvolutionOp, self).__init__(args=(inputs, filters), **kwargs)
        else:
            super(ConvolutionOp, self).__init__(args=(inputs, filters, bias), **kwargs)

        if len(inputs.shape) != 5:
            raise ValueError((
                'convolution input shape must be length 5, found {}'
            ).format(len(inputs.shape)))

        if len(filters.shape) != 5:
            raise ValueError((
                'convolution filter shape must be length 5, found {}'
            ).format(len(filters.shape)))

        if not inputs.axes[0] == filters.axes[0]:
            raise ValueError((
                'the first axis in input {inputs} and filter {filters} are not the same.'
            ).format(inputs=inputs.axes[0], filters=filters.axes[0]))

        expected_keys = ['pad_h', 'pad_w', 'pad_d', 'str_h', 'str_w',
                         'str_d', 'dil_h', 'dil_w', 'dil_d']
        # TODO: meybe we should assume no padding and no dilitation when
        # these parameters are not given
        for k in expected_keys:
            if k not in conv_params:
                raise ValueError((
                    'Expected parameter {key} not present in convparams dict.'
                ).format(key=k))

        self.conv_params = conv_params
        self.__has_side_effects = False

    def copy_with_new_args(self, args):
        return type(self)(self.conv_params, *args, axes=self.axes)

    def generate_adjoints(self, adjoints, delta, inputs, filters, bias=None):
        """
        TODO
        """
        # requires conv's forward to be completed before backward:
        update_conv_op = update_conv(delta, inputs, filters, self)
        update_conv_op.add_control_dep(self)
        bprop_conv_op = bprop_conv(delta, inputs, filters, self)
        bprop_conv_op.add_control_dep(self)
        filters.generate_add_delta(adjoints, update_conv_op)
        inputs.generate_add_delta(adjoints, bprop_conv_op)

    @property
    def has_side_effects(self):
        return self.__has_side_effects

    @has_side_effects.setter
    def has_side_effects(self, value):
        self.__has_side_effects = value


def deconvolution(conv_params, inputs, filters, axes, docstring=None):
    """

    Args:
        conv_params: Dimensions.
        inputs (TensorOp): The input tensor.
        filters (TensorOp): Filter/kernel tensor.
        docstring (String, optional): Documentation for the op.

    Returns:
        TensorOp: The result of the deconvolution.
    """
    return DeconvolutionOp(conv_params, inputs, filters, axes=axes, docstring=docstring)


class DeconvolutionOp(TensorOp):
    """
    Arguments:
        inputs  : input tensor.
        filters : filter/kernel tensor.

    Return:
    """

    def __init__(self, conv_params, inputs, filters, **kwargs):
        super(DeconvolutionOp, self).__init__(args=(inputs, filters), **kwargs)

        if len(inputs.shape) != 5:
            raise ValueError((
                'convolution input shape must be length 5, found {}'
            ).format(len(inputs.shape)))

        if len(filters.shape) != 5:
            raise ValueError((
                'convolution filter shape must be length 5, found {}'
            ).format(len(filters.shape)))

        if not inputs.axes[0] == filters.axes[0]:
            raise ValueError((
                'the first axis in input {inputs} and filter {filters} are not the same.'
            ).format(inputs=inputs.axes[0], filters=filters.axes[0]))

        expected_keys = ['pad_h', 'pad_w', 'pad_d', 'str_h', 'str_w',
                         'str_d', 'dil_h', 'dil_w', 'dil_d']
        # TODO: maybe we should assume no padding and no dilation when
        # these parameters are not given
        for k in expected_keys:
            if k not in conv_params:
                raise ValueError((
                    'Expected parameter {key} not present in convparams dict.'
                ).format(key=k))

        self.conv_params = conv_params
        self.__has_side_effects = False

    @property
    def has_side_effects(self):
        return self.__has_side_effects

    @has_side_effects.setter
    def has_side_effects(self, value):
        self.__has_side_effects = value

    def copy_with_new_args(self, args):
        return type(self)(self.conv_params, *args, axes=self.axes)

    def generate_adjoints(self, adjoints, delta, inputs, filters):
        # requires conv's forward to be completed before backward
        update_conv_op = update_conv(inputs, delta, filters, self)  # switch inputs and delta
        update_conv_op.add_control_dep(self)
        deconv_deriv_op = DeconvDerivOp(delta, inputs, filters, self)
        deconv_deriv_op.add_control_dep(self)
        filters.generate_add_delta(adjoints, update_conv_op)
        inputs.generate_add_delta(adjoints, deconv_deriv_op)


class ConvDerivOp(TensorOp):
    """
    Maintains index and conv_params through forwarding of the original convolution.

    Arguments:
        fprop: The original convolution.
    """
    def __init__(self, fprop, **kwargs):
        super(ConvDerivOp, self).__init__(**kwargs)
        self.fprop = fprop
        fprop.has_side_effects = True

    @property
    def conv_params(self):
        """

        Returns:
            The convolution parameters of the convolution.

        """
        return self.fprop.forwarded.conv_params


class update_conv(ConvDerivOp):
    """
    Arguments:
        inputs  : input tensor.
        filters : filter/kernel tensor.
    """
    def __init__(self, delta, inputs, filters, fprop, **kwargs):
        super(update_conv, self).__init__(
            args=(delta, inputs),
            fprop=fprop,
            axes=filters.axes, **kwargs
        )

    def copy_with_new_args(self, args):
        return type(self)(args[0], args[1], self.fprop.args[1], self.fprop)


class bprop_conv(ConvDerivOp):
    """
    Arguments:
        inputs  : input tensor.
        filters : filter/kernel tensor.
    """
    def __init__(self, delta, inputs, filters, fprop, **kwargs):
        super(bprop_conv, self).__init__(
            args=(delta, filters),
            fprop=fprop,
            axes=inputs.axes, **kwargs
        )

    def copy_with_new_args(self, args):
        return type(self)(args[0], self.fprop.args[0], args[1], self.fprop)


class DeconvDerivOp(ConvDerivOp):
    def __init__(self, delta, inputs, filters, fprop, **kwargs):
        """
        Deconv backprop

        Arguments:
            inputs  : input tensor.
            filters : filter/kernel tensor.
        """
        super(DeconvDerivOp, self).__init__(
            args=(delta, filters),
            fprop=fprop,
            axes=inputs.axes, **kwargs
        )

    def copy_with_new_args(self, args):
        return type(self)(args[0], self.fprop.args[0], args[1], self.fprop)
