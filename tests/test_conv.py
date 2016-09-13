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

import ngraph as ng
from ngraph.util.derivative_check import check_derivative
from ngraph.util.utils import executor
from ngraph.util.utils import RandomTensorGenerator

rng = RandomTensorGenerator(0, np.float32)


def test_convolution_axis():
    """ convolution with stride 1 and width 1 returns the same size axis """
    a = ng.Axis(5)
    assert ng.ConvolutionAxis(a, 1, 1) == a


def test_convolution_axis_derivative_match():
    """
    the derivative of a convolution wrt the input should have the same axes
    as the input
    """
    a = ng.Axis(5)
    conv = ng.ConvolutionAxis(a, 5, 1)
    pad = ng.PaddedAxis(conv, (4, 4))
    out = ng.ConvolutionAxis(pad, 5, 1)

    assert out.length == a.length
    assert out == a


def np_convolution1d(input, filter):
    """ numpy implementation of convolution1d with stride 1 and no padding """

    # add Cout and N dimensions to result_np
    result_np = np.zeros((
        filter.shape[2],
        input.shape[1] - filter.shape[1] + 1,
        filter.shape[2]
    ), dtype='float32')

    for i in range(filter.shape[2]):
        result_np[i, :, 0] = np.convolve(
            input[0, :, 0], filter[0, ::-1, 0], mode='valid'
        )

    return result_np


def test_constant_tensor_convolution_fprop_1d():
    """
    A simple test running a convolution filter over an input where both filter
    and input are ones and both are the same size.
    """

    N = ng.Axis(1, batch=True)
    T = ng.Axis(5)
    FT = ng.Axis(3)
    Cin = ng.Axis(1)
    Cout = ng.Axis(1)

    input = ng.placeholder(axes=ng.Axes([Cin, T, N]))
    filter = ng.placeholder(axes=ng.Axes([Cin, FT, Cout]))

    # randomly initialize
    input_value = rng.uniform(-1, 1, input.axes)
    filter_value = rng.uniform(-1, 1, filter.axes)

    # initialize with specific, easy to follow values
    input_value = np.array([[[0], [0], [0], [1], [0]]])
    filter_value = np.array([[[0], [1], [2]]])

    assert input_value.shape == (Cin.length, T.length, N.length)
    assert filter_value.shape == (Cin.length, FT.length, Cout.length)

    # compute convolution with graph
    output = ng.convolution1d(input, filter)
    result_og = executor(output, input, filter)(input_value, filter_value)

    result_np = np_convolution1d(input_value, filter_value)

    np.testing.assert_allclose(result_og, result_np)


def check_constant_tensor_convolution_bprop_input_1d(
    N=None, T=None, FT=None, Cin=None, Cout=None
):
    """
    A simple test running a convolution filter over an input
    """

    N = ng.Axis(N, name='N', batch=True)
    T = ng.Axis(T, name='T')
    FT = ng.Axis(FT, name='FT')
    Cin = ng.Axis(Cin, name='Cin')
    Cout = ng.Axis(Cout, name='Cout')

    input = ng.placeholder(axes=ng.Axes([Cin, T, N]))
    filter = ng.placeholder(axes=ng.Axes([Cin, FT, Cout]))

    # randomly initialize
    input_value = rng.uniform(-1, 1, input.axes)
    filter_value = rng.uniform(-1, 1, filter.axes)

    check_derivative(
        ng.convolution1d(input, filter),
        input, 0.001, input_value,
        [filter], [filter_value],
        atol=1e-3, rtol=1e-3
    )


def test_convolution_1d_bprop_simple():
    check_constant_tensor_convolution_bprop_input_1d(
        N=1, T=2, FT=1, Cin=1, Cout=1,
    )


def test_convolution_1d_bprop_wider_filter():
    check_constant_tensor_convolution_bprop_input_1d(
        N=1, T=10, FT=5, Cin=1, Cout=1,
    )


def test_convolution_1d_bprop_multibatch():
    check_constant_tensor_convolution_bprop_input_1d(
        N=128, T=2, FT=1, Cin=1, Cout=1,
    )


def test_convolution_1d_bprop_multi_channel():
    # cin != cout
    check_constant_tensor_convolution_bprop_input_1d(
        N=1, T=2, FT=1, Cin=16, Cout=5,
    )


def test_convolution_1d_bprop_complex():
    # all dimensions are non-1; cin != cout
    check_constant_tensor_convolution_bprop_input_1d(
        N=3, T=7, FT=5, Cin=4, Cout=6,
    )
