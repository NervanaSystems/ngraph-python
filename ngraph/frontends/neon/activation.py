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
from builtins import object
import ngraph as ng


class Rectlin(object):
    """
    Rectified Linear Unit (ReLu) activation function, :math:`f(x) = \max(x, 0)`.
    Can optionally set a slope which will make this a Leaky ReLu.
    """

    def __init__(self, slope=0, **kwargs):
        """
        Class constructor.

        Arguments:
            slope (float, optional): Slope for negative domain. Defaults to 0.
        """
        self.slope = slope

    def __call__(self, x):
        """
        Returns the Exponential Linear activation

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: output activation
        """
        return ng.maximum(x, 0) + self.slope * ng.minimum(0, x)


class Identity(object):
    """Identity activation function, :math:`f(x) = x`"""

    def __call__(self, x):
        """
        Returns the input as output.

        Arguments:
            x (Tensor or optree): input value

        Returns:
            Tensor or optree: identical to input
        """
        return x


class Explin(object):
    """
    Exponential Linear activation function, :math:`f(x) = \max(x, 0) + \\alpha (e^{\min(x, 0)}-1)`
    From: Clevert, Unterthiner and Hochreiter, ICLR 2016.
    """

    def __init__(self, alpha=1.0):
        """
        Class constructor.

        Arguments:
            alpha (float): weight of exponential factor for negative values (default: 1.0).
            name (string, optional): Name (default: None)
        """
        self.alpha = alpha

    def __call__(self, x):
        """
        Returns the Exponential Linear activation

        Arguments:
            x (Tensor or optree): input value

        Returns:
            Tensor or optree: output activation
        """
        return ng.maximum(x, 0) + self.alpha * (ng.exp(ng.minimum(x, 0)) - 1)


class Normalizer(object):
    """Normalize inputs by a fixed divisor."""

    def __init__(self, divisor=128.):
        """
        Class constructor.

        Arguments:
            divisor (float, optional): Normalization factor (default: 128)
            name (string, optional): Name (default: None)
        """
        self.divisor = divisor

    def __call__(self, x):
        """
        Returns the normalized value.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: Output :math:`x / N`
        """
        return x / self.divisor


class Softmax(object):
    """SoftMax activation function. Ensures that the activation output sums to 1."""

    def __call__(self, x):
        """
        Returns the Softmax value.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: Output activation
        """
        return ng.softmax(x)


class Tanh(object):
    """Hyperbolic tangent activation function, :math:`f(x) = \\tanh(x)`."""

    def __call__(self, x):
        """
        Returns the hyperbolic tangent.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: Output activation
        """
        return ng.tanh(x)


class Logistic(object):
    """
    Logistic sigmoid activation function, :math:`f(x) = 1 / (1 + \exp(-x))`

    Squashes the input from range :math:`[-\infty,+\infty]` to :math:`[0, 1]`
    """

    def __call__(self, x):
        """
        Returns the sigmoidal activation.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: Output activation
        """
        return ng.sigmoid(x)
