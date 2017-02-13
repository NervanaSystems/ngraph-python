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

from __future__ import division
import ngraph as ng
import math


def common_conv2d_pool_output_shape(in_NHWC, f_HWIO, str_NHWC, padding):
    """
    Get tensorflow's tf.nn.conv2d output shape
    TODO: currently only support NHWC * RSCK, to support NCHW.

    Args:
        in_NHWC: [batch, in_height, in_width, in_channels].
        f_HWIO: [filter_height, filter_width, in_channels, out_channels].
        str_NHWC: List of ints of length 4.
        padding: A string from: "SAME", "VALID".

    Returns:
        output shape of tf.nn.conv2d
    """
    # check inputs
    if padding != 'SAME' and padding != 'VALID':
        raise ValueError("Padding must be 'SAME' or 'valid'.")
    if not (len(in_NHWC) == len(f_HWIO) == len(str_NHWC) == 4):
        raise ValueError(
            "in_NHWC, f_HWIO, str_NHWC must be length 4.")

    # get input / filter shape
    N, H, W, C = in_NHWC
    R, S, C_, K = f_HWIO
    if C != C_:
        raise ValueError("Input channel must be the same as filter channel.")

    # only support [1, X, X, 1] str_NHWC for importer now
    if str_NHWC[0] != 1 or str_NHWC[3] != 1:
        raise NotImplementedError("Strides on batch axis (N) and channel axis "
                                  "(C) must be 1 for importer.")

    # get output shape
    if 'SAME' == padding:
        out_height = math.ceil(float(H) / float(str_NHWC[1]))
        out_width = math.ceil(float(W) / float(str_NHWC[2]))
    elif 'VALID' == padding:
        out_height = math.ceil(float(H - R + 1) / float(str_NHWC[1]))
        out_width = math.ceil(float(W - S + 1) / float(str_NHWC[2]))

    return tuple([int(i) for i in N, out_height, out_width, K])


def common_conv2d_pool_padding(in_NHWC, f_HWIO, str_NHWC, padding):
    """
    Get tensorflow's tf.nn.conv2d padding size
    TODO: currently only support NHWC * RSCK, to support NCHW.

    Args:
        in_NHWC: [batch, in_height, in_width, in_channels].
        f_HWIO: [filter_height, filter_width, in_channels, out_channels].
        str_NHWC: List of ints of length 4.
        padding: A string from: "SAME", "VALID".

    Returns:
        pad_top, pad_bottom, pad_left, pad_right
    """
    # check validity and get output size
    _, out_height, out_width, _ = common_conv2d_pool_output_shape(
        in_NHWC, f_HWIO, str_NHWC, padding)
    if padding == 'SAME':
        # get input / filter shape
        N, H, W, C = in_NHWC
        R, S, C_, K = f_HWIO

        # get padding size
        pad_along_height = ((out_height - 1) * str_NHWC[1] + R - H)
        pad_along_width = ((out_width - 1) * str_NHWC[2] + S - W)
        pad_top = int(pad_along_height) // 2
        pad_bottom = int(pad_along_height - pad_top)
        pad_left = int(pad_along_width) // 2
        pad_right = int(pad_along_width - pad_left)
        return (pad_top, pad_bottom, pad_left, pad_right)
    else:
        return (0, 0, 0, 0)


class CommonSGDOptimizer(object):
    def __init__(self, lrate=0.1):
        self.lrate = lrate

    def minimize(self, cost, variables):
        """
        Minimize cost by returning update Ops.

        Arguments:
            cost: The cost Op to be minimized
            variables: TODO

        Returns:
            A doall op containing setitems to variable ops.
        """

        assert cost is not None
        assert variables is not None

        return ng.doall((ng.assign(variable, variable - self.lrate * ng.deriv(cost, variable))
                         for variable in variables))
