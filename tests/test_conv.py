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

import numpy as np
# import pytest

import nervanagraph as ng
from nervanagraph.op_graph import arrayaxes
import nervanagraph.frontends.base.axis as ax
from nervanagraph.util.utils import executor


def test_constant_tensor_convolution_fprop():
    """
    A simple test running a convolution filter over an input where both filter
    and input are ones and both are the same size.
    """

    ax.N.length = 1
    ax.C.length = 2
    ax.H.length = 2
    ax.W.length = 2
    ax.Cout = arrayaxes.Axis(2)

    input_axes = arrayaxes.Axes([ax.C, ax.H, ax.W, ax.N])
    filter_axes = arrayaxes.Axes([ax.C, ax.H, ax.W, ax.Cout])

    input = ng.Constant(
        np.ones(input_axes.lengths, dtype='float32'), axes=input_axes,
    )
    filter = ng.Constant(
        np.ones(filter_axes.lengths, dtype='float32'), axes=filter_axes,
    )

    output = ng.convolution(input, filter)

    result = executor(output)()
    assert np.allclose(result, [[[8.0]]])


def test_constant_tensor_convolution_deriv():
    """
    A simple test running a convolution filter over an input where both filter
    and input are ones and both are the same size.
    """

    ax.N.length = 1
    ax.C.length = 3
    ax.H.length = 5
    ax.W.length = 5
    ax.FilterH = arrayaxes.Axis(3)
    ax.FilterW = arrayaxes.Axis(3)
    ax.Cout = arrayaxes.Axis(3)

    input_axes = arrayaxes.Axes([ax.C, ax.H, ax.W, ax.N])
    filter_axes = arrayaxes.Axes([ax.C, ax.FilterH, ax.FilterW, ax.Cout])

    input = ng.Variable(axes=input_axes)
    filter = ng.Variable(axes=filter_axes)

    output = ng.convolution(input, filter, padding=[1, 1, 1, 1])

    ng.deriv(output, input)
