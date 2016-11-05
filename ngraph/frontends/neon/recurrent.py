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
import ngraph as ng
from ngraph.frontends.neon.layer import ParameterLayer
import numpy as np


def get_steps(x, time_axis):
    steps = []
    for i in range(time_axis.length):
        s_axis = ng.slice_along_axis(x, time_axis, i)
        steps.append(s_axis)
    return steps


class Recurrent(ParameterLayer):
    """
    Basic recurrent layer.
    Arguments:
        output_size (int): Number of hidden/output units
        init (Initializer): Function for initializing the model's input to hidden weights.  By
                            default, this initializer will also be used for recurrent parameters
                            unless init_inner is also specified.  Biases will always be
                            initialized to zero.
        init_inner (Initializer, optional): Function for initializing the model's recurrent
                                            parameters.  If absent, will default to using same
                                            initializer provided to init.
        activation (Transform): Activation function for the input modulation
        reset_cells (bool): default to be False to make the layer stateful,
                            set to True to be stateless
        name (str, optional): name to refer to this layer as.
    Attributes:
        W_input (Tensor): weights from inputs to output units
            (input_size, output_size)
        W_recur (Tensor): weights for recurrent connections
            (output_size, output_size)
        b (Tensor): Biases on output units (output_size, 1)
    """

    def __init__(self, output_size, init, init_inner=None, activation=None,
                 time_axis=None, **kwargs):
        super(Recurrent, self).__init__(init=init, **kwargs)
        self.nout = output_size
        self.h_out = output_size
        self.activation = activation
        self.init_inner = init_inner or init
        self.time_axis = time_axis

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
           in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                 information for layer

        Returns:
           (Tensor): output

        """
        in_obj = super(Recurrent, self).configure(in_obj)

        if self.axes is not None:
            hidden_axes = self.axes - self.axes.recurrent_axes()
        else:
            hidden_axes = ng.make_axes([ng.make_axis(self.nout, name='Hidden_in')])

        self.W_input = ng.variable(
            hidden_axes + (in_obj.axes.sample_axes() - in_obj.axes.recurrent_axes()).get_dual(),
            init=self.init,
            name="W_input"
        )

        self.W_recur = ng.variable(
            hidden_axes + [axis - 1 for axis in hidden_axes],
            init=self.init_inner,
            name="W_recur"
        )

        self.b = ng.variable(
            hidden_axes,
            initial_value=0,
            name="bias"
        )

        h_ff_buf = ng.dot(self.W_input, in_obj, use_dual=True, name="W_in_dot_in")
        h_ff_s = get_steps(h_ff_buf, self.time_axis)
        self.h_init = ng.constant(np.zeros(h_ff_s[0].axes.lengths),
                                  h_ff_s[0].axes,
                                  name="h_init")
        hprev = [self.h_init]

        for i in range(self.time_axis.length):
            d = ng.dot(self.W_recur, hprev[i], use_dual=True, name="W_rec_dot_h{}".format(i))
            h = self.activation(d + h_ff_s[i] + self.b)
            h.name = "activ{}".format(i)
            hprev.append(h)

        rnn_out = ng.Stack(hprev[1:], self.time_axis, pos=1)
        return rnn_out
