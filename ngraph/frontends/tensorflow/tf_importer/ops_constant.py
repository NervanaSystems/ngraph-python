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
from ngraph.frontends.tensorflow.tf_importer.ops_base import OpsBase
from ngraph.frontends.tensorflow.tf_importer.utils import shape_to_axes
from tensorflow.python.framework import tensor_util
import ngraph as ng
import numpy as np
import scipy.stats


class OpsConstant(OpsBase):
    """
    Mix-in class for unary ops.
    """

    def Const(self, tf_node, inputs):
        """
        Creates a constant tensor.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            value, dtype, shape, name
        """
        # convert to numpy value
        np_val = tensor_util.MakeNdarray(tf_node.attr['value'].tensor)
        ng_op = ng.constant(
            np_val, axes=shape_to_axes(np_val.shape), name=tf_node.name)
        return ng_op

    def Fill(self, tf_node, inputs):
        """
        Creates a tensor filled with a scalar value.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            dims, value, name
        """
        # get inputs
        shape_op, const_val_op = inputs

        # get shape, const_val
        shape = tuple(shape_op.const.astype(int))
        const_val = const_val_op.const

        # convert to numpy value
        np_val = np.zeros(shape)
        np_val.fill(const_val)

        # create op
        ng_op = ng.constant(
            np_val, axes=shape_to_axes(np_val.shape), name=tf_node.name)
        return ng_op

    def TruncatedNormal(self, tf_node, inputs):
        """
        Outputs random values from a truncated normal distribution.
        `tf.truncated_normal()` call generates several ops, the
        The `TruncatedNormal` op is what we implement here.

        shape --> TruncatedNormal
                       |
                       V
        stddev -----> Mul
                       |
                       V
        mean -------> Add
                       |
                       V
                    (output)

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            shape, mean, dtype, seed, name
        """
        # get inputs
        shape = tuple(inputs[0].const.astype(int))

        # generate truncated standard normal
        mu, sigma, lo, up = 0., 1., -2., 2
        generator = scipy.stats.truncnorm(
            (lo - mu) / sigma, (up - mu) / sigma, loc=mu, scale=sigma)
        np_val = generator.rvs(shape)
        ng_op = ng.constant(
            np_val, axes=shape_to_axes(np_val.shape), name=tf_node.name)
        return ng_op

    def RandomStandardNormal(self, tf_node, inputs):
        """
        Outputs random values from a normal distribution. `tf.random_normal()`
        call generates several ops. The `RandomStandardNormal` op is what we
        implement here.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            shape, mean, dtype, seed, name
        """
        # get inputs
        shape = tuple(inputs[0].const.astype(int))

        # generate standard normal
        np_val = np.random.standard_normal(size=shape)
        ng_op = ng.constant(
            np_val, axes=shape_to_axes(np_val.shape), name=tf_node.name)
        return ng_op

    def ZerosLike(self, tf_node, inputs):
        """
        Creates a tensor with all elements set to zero.

        Returns:
            A `Tensor` with all elements set to zero.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            tensor, dtype, name
        """
        shape = inputs[0].axes.lengths
        np_val = np.zeros(shape)
        ng_op = ng.constant(
            np_val, axes=shape_to_axes(np_val.shape), name=tf_node.name)
        return ng_op
