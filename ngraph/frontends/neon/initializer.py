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
from functools import partial
from ngraph.frontends.neon.axis import is_shadow_axis


class GaussianInit(object):

    def __init__(self, mean=0.0, var=0.01):
        self.functor = partial(np.random.normal, mean, var)

    def __call__(self, w_axes):
        return self.functor(w_axes.lengths)


class UniformInit(object):

    def __init__(self, low=-0.01, high=0.01):
        self.functor = partial(np.random.uniform, low, high)

    def __call__(self, w_axes):
        return self.functor(w_axes.lengths)


class ConstantInit(object):

    def __init__(self, val=0.0):
        self.val = val

    def __call__(self, w_axes):
        return self.val


def _input_output_axes(w_axes):
    """
    Given the axes from a tensor of weights, provides the axes corresponding to inputs
    (often called 'fan-in') and the axes corresponding to outputs (often called 'fan-out').

    Args:
        w_axes (Axes): Axes of weight tensor

    Returns:
        axes_i (Axes): Fan-in axes
        axes_o (Axes): Fan-out axes

    Note:
        Assumes that output axes are shadow axes
    """

    return (
        ng.make_axes([axis for axis in w_axes if not is_shadow_axis(axis)]),
        ng.make_axes([axis for axis in w_axes if is_shadow_axis(axis)]),
    )


class GlorotInit(object):

    def __call__(self, w_axes):
        input_axes, output_axes = _input_output_axes(w_axes)
        scale = np.sqrt(6. / (np.prod(input_axes.lengths) + np.prod(output_axes.lengths)))
        return np.random.uniform(-scale, scale, w_axes.lengths)


class XavierInit(object):

    def __call__(self, w_axes):
        input_axes, _ = _input_output_axes(w_axes)
        scale = np.sqrt(3. / np.prod(input_axes.lengths))
        return np.random.uniform(-scale, scale, w_axes.lengths)


class KaimingInit(object):

    def __call__(self, w_axes):
        input_axes, _ = _input_output_axes(w_axes)
        scale = np.sqrt(2. / np.prod(input_axes.lengths))
        return np.random.normal(0, scale, w_axes.lengths)
